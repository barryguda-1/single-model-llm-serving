from typing import List
from .workload_manager import WorkloadManager, Sequence
from .model_executor import ModelExecutor
import asyncio
import json
import atexit
import logging
import threading
import uuid
from vllm import LLM as VLLM
from vllm import SamplingParams

logger = logging.getLogger(__name__)

class LLMEngine:
    def __init__(self, model_name: str = "facebook/opt-125m", max_tokens: int = 20):
        self.model_executor = ModelExecutor()
        self.workload_manager = WorkloadManager()
        self.max_tokens = max_tokens

        # Initialize the model worker (HF transformers, separate process)
        self.model_executor.setup_worker(model_name)

        # Initialize vLLM model
        self.vllm_model = VLLM(model=model_name)

        # Start processing loop in a separate thread
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self.requests_processing_loop, daemon=True)
        self.thread.start()

        # Register cleanup
        atexit.register(self._cleanup)

    def requests_processing_loop(self):
        """Continuous-batching loop for streaming requests (runs in its own thread)."""
        while not self._stop_event.is_set():
            active_sequences = []
            try:
                active_sequences = self.workload_manager.get_next_batch(is_streaming=True)
                if not active_sequences:
                    self._stop_event.wait(0.1)
                    continue

                # Process batch through model, forward pass.
                prompts = [{'prompt': seq.prompt, 'request_id': seq.id} for seq in active_sequences]
                prompts_results = self.model_executor.execute_forward_batch(prompts)

                for result in prompts_results:
                    seq = self.workload_manager.get_sequence(result['request_id'])
                    if seq is None:
                        # Sequence was already removed (e.g. client disconnected).
                        continue

                    # Update first, then check the limit (keeps token_count exact).
                    self.workload_manager.update_sequence_output(
                        result['request_id'],
                        result['token'],
                        is_finished=result['is_finished'],
                    )

                    # EOS tokens are not streamed to the client.
                    if not result['is_finished']:
                        # Use run_coroutine_threadsafe to safely put data in the main loop's queue
                        asyncio.run_coroutine_threadsafe(
                            seq.client_stream.put(
                                json.dumps({"token": result['token'], "sequence_id": result['request_id']})
                            ),
                            seq.loop
                        )

                    if result['is_finished'] or seq.token_count >= self.max_tokens:
                        seq.finished = True
                        # None = end-of-stream sentinel for the client.
                        asyncio.run_coroutine_threadsafe(
                            seq.client_stream.put(None),
                            seq.loop
                        )
                        self.workload_manager.remove_finished_sequence(result['request_id'])

            except Exception:
                logger.exception("Error in processing loop")
                # Never leave clients hanging: end every stream in the failed batch.
                for seq in active_sequences:
                    try:
                        asyncio.run_coroutine_threadsafe(seq.client_stream.put(None), seq.loop)
                    except Exception:
                        pass
                self._stop_event.wait(0.1)

    def _cleanup(self):
        """Stop the processing loop and shut down the worker process."""
        self._stop_event.set()
        self.thread.join(timeout=5)
        self.model_executor.shutdown()

    # process 1 request with only one prompt at a time.
    def basic_generate(self, prompt: str) -> str:
        return self.generate([prompt])[0]

    # process multiple prompts in a request, in fixed-size batches.
    # Sequences are local to this call, so the sync path cannot interfere
    # with the shared streaming pool in the workload manager.
    def generate(self, prompts: List[str]) -> List[str]:
        generated_texts = []
        batch_size = self.workload_manager.batch_size
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            sequences = [Sequence(str(uuid.uuid4()), prompt, None, None) for prompt in chunk]

            # Execute the batch (the worker preserves order within a batch)
            results = self.model_executor.execute_batch(sequences)
            generated_texts.extend(result['generated_text'] for result in results)

        return generated_texts

    async def event_generator(self, loop, prompt: str):
        # Create a queue for this client's stream
        queue = asyncio.Queue()

        # Add streaming request to workload manager with the queue
        seq_id = self.workload_manager.add_streaming_request(prompt, queue, loop)

        try:
            while True:
                # Get next token from the queue (None = end of stream)
                data = await queue.get()
                if data is None:
                    break
                yield f"data: {data}\n\n"
        except Exception:
            logger.exception(f"Error in stream for sequence {seq_id}")
        finally:
            # Clean up (idempotent — the processing loop may have removed it already)
            self.workload_manager.remove_finished_sequence(seq_id)

    def generate_vllm(self, prompts: List[str]) -> List[str]:
        """
        Generate text using vLLM for multiple prompts.

        Args:
            prompts: List of prompts to generate text for

        Returns:
            List of generated texts
        """
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=self.max_tokens
        )

        # Generate text for all prompts
        outputs = self.vllm_model.generate(prompts, sampling_params)

        # Extract generated text from outputs
        generated_texts = [output.outputs[0].text for output in outputs]

        return generated_texts
