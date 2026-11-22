from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llm import LLMEngine
from typing import List
import asyncio
import os
import threading
import atexit
import signal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Create LLM instance
_llm = None
_llm_lock = threading.Lock()

def cleanup():
    global _llm
    if _llm is not None:
        try:
            _llm._cleanup()
        except Exception:
            logger.exception("Error during LLM engine cleanup")
        _llm = None

def get_llm():
    global _llm
    with _llm_lock:
        if _llm is None:
            _llm = LLMEngine(
                model_name=os.environ.get("MODEL_NAME", "facebook/opt-125m"),
                max_tokens=int(os.environ.get("MAX_TOKENS", "20")),
            )
            # Register cleanup
            atexit.register(cleanup)
        return _llm

class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    generated_text: str

class BatchGenerateRequest(BaseModel):
    prompts: List[str]

class BatchGenerateResponse(BaseModel):
    generated_texts: List[str]

@app.post("/generate_stream")
async def generate_stream(request: GenerateRequest, llm: LLMEngine = Depends(get_llm)):
    async def event_generator():
        loop = asyncio.get_event_loop()
        async for token in llm.event_generator(loop, request.prompt):
            yield token

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# The endpoints below call the engine synchronously (blocking queue waits and
# full generations), so they are plain `def`: FastAPI runs them in its
# threadpool instead of freezing the event loop for every other request.

# process 1 request with only one prompt at a time.
@app.post("/basic_generate", response_model=GenerateResponse)
def basic_generate(request: GenerateRequest, llm: LLMEngine = Depends(get_llm)):
    generated_text = llm.basic_generate(request.prompt)
    return GenerateResponse(generated_text=generated_text)

# process multiple prompts in a request
@app.post("/generate", response_model=BatchGenerateResponse)
def generate(request: BatchGenerateRequest, llm: LLMEngine = Depends(get_llm)):
    generated_texts = llm.generate(request.prompts)
    return BatchGenerateResponse(generated_texts=generated_texts)

@app.post("/generate_vllm", response_model=BatchGenerateResponse)
def generate_vllm(request: BatchGenerateRequest, llm: LLMEngine = Depends(get_llm)):
    """
    Generate text using vLLM for multiple prompts.
    This endpoint uses vLLM's efficient batched inference capabilities.
    """
    generated_texts = llm.generate_vllm(request.prompts)
    return BatchGenerateResponse(generated_texts=generated_texts)

def signal_handler(signum, frame):
    cleanup()
    exit(0)

if __name__ == "__main__":
    import uvicorn
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize LLM before starting the server
    get_llm()

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
