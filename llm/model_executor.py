import multiprocessing as mp
import threading
from typing import List, Dict, Any
from .model_worker import ModelWorker
import logging
import sys

# Seconds to wait for the worker before declaring a batch failed
RESULT_TIMEOUT_S = 60

# Set up logging with stream handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class ModelExecutor:
    def __init__(self):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.worker_process = None
        # Serializes full put+get round-trips so the streaming-loop thread
        # and request threads can never consume each other's results.
        self._lock = threading.Lock()
        logger.debug("ModelExecutor initialized with queues")

    def setup_worker(self, model_name: str):
        logger.debug(f"Setting up worker with model: {model_name}")
        self.worker_process = mp.Process(
            target=ModelWorker.run,
            args=(model_name, self.task_queue, self.result_queue),
            daemon=True,
        )
        logger.debug("Starting worker process")
        self.worker_process.start()
        logger.debug("Worker process started")

    def execute_batch(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not prompts:
            logger.debug("Empty batch received")
            return []

        logger.debug(f"Sending batch to worker: {prompts}")
        with self._lock:
            # Send batch to worker
            self.task_queue.put((prompts, False))

            # Get results
            logger.debug("Waiting for results from worker")
            result_type, results = self.result_queue.get(timeout=RESULT_TIMEOUT_S)

        logger.debug(f"Received results from worker: {results}")
        if result_type == 'complete':
            return results
        raise RuntimeError(f"Unexpected result type from worker: {result_type}")

    def execute_forward_batch(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not prompts:
            logger.debug("Empty batch received")
            return []

        logger.debug(f"Sending streaming batch to worker: {prompts}")
        with self._lock:
            # Send batch to worker with streaming flag
            self.task_queue.put((prompts, True))

            # Get streaming results
            logger.debug("Waiting for streaming results from worker")
            result_type, results = self.result_queue.get(timeout=RESULT_TIMEOUT_S)

        logger.debug(f"Received streaming results from worker: {results}")

        if result_type == 'stream':
            return results
        raise RuntimeError(f"Unexpected result type from worker: {result_type}")

    def shutdown(self):
        """Ask the worker to exit and wait for it (falls back to terminate)."""
        if self.worker_process is None or not self.worker_process.is_alive():
            return
        logger.debug("Sending shutdown signal to worker")
        try:
            # None is the shutdown sentinel the worker waits for
            self.task_queue.put(None)
            self.worker_process.join(timeout=10)
        except Exception:
            logger.exception("Error while shutting down worker")
        if self.worker_process.is_alive():
            logger.warning("Worker did not exit cleanly, terminating")
            self.worker_process.terminate()
            self.worker_process.join()
        logger.debug("Worker process shut down")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
