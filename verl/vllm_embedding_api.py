#!/usr/bin/env python3
"""
vLLM Embedding API Service
HTTP interface for embedding service based on FastAPI and vLLM >= 0.8.5
"""

import torch
import vllm
from vllm import LLM
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import argparse
import asyncio
import time
import threading

# Request/Response models
class EmbeddingRequest(BaseModel):
    texts: List[str]
    task_description: Optional[str] = None

class SimilaritySearchRequest(BaseModel):
    queries: List[str]
    documents: List[str]
    task_description: str = 'Given a web search query, retrieve relevant passages that answer the query'

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    processing_time: float

class SimilaritySearchResponse(BaseModel):
    scores: List[List[float]]
    processing_time: float

class VLLMEmbeddingAPI:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B", 
                 tensor_parallel_size: int = 8,
                 gpu_memory_utilization: float = 0.8):
        """
        Initialize vLLM embedding API service
        """
        print(f"Initializing vLLM embedding API service, model: {model_name}")
        
        self.model = LLM(
            model=model_name,
            task="embed",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="auto",
            enforce_eager=False,  # Use CUDA graph acceleration, consistent with verl
            enable_prefix_caching=False,  # 🔥 Disable prefix caching to prevent memory leaks
            max_model_len=16384,  # Reduce max sequence length to improve speed
            swap_space=2,  # Reduce swap space
            disable_log_stats=True,  # Disable statistics logging to improve performance
            max_num_seqs=256,  # Increase max sequence count, improve batch processing capability
            max_num_batched_tokens=1000000,  # Reduce batch token count to avoid excessive memory usage
            disable_custom_all_reduce=True,  # Key: disable custom all_reduce, consistent with verl
            distributed_executor_backend="mp",  # Use multiprocessing backend
        )
        
        print("vLLM embedding API service initialization completed")
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Generate detailed instruction"""
        return f'Instruct: {task_description}\nQuery:{query}'
    
    async def embed_texts_batch(self, texts: List[str]) -> torch.Tensor:
        """Batch embed texts - true batch processing implementation"""
        # Create Future to wait for results
        future = asyncio.Future()

        # Create batch request
        batch_request = BatchRequest(texts, future)

        # Add to batch queue
        with batch_lock:
            batch_queue.append(batch_request)
            batch_event.set()  # Notify batch processor

        # Wait for batch processing results
        return await future

    def embed_texts_sync(self, texts: List[str]) -> torch.Tensor:
        """Synchronous text embedding - kept for compatibility"""
        try:
            outputs = self.model.embed(texts)
            embeddings = torch.stack([torch.tensor(o.outputs.embedding) for o in outputs])
            return embeddings
        except Exception as e:
            print(f"❌ vLLM processing error: {str(e)}")
            raise

    async def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Asynchronous text embedding"""
        try:
            # Run CPU-intensive operations in thread pool
            loop = asyncio.get_event_loop()

            def _embed():
                try:
                    print(f"🔄 vLLM processing {len(texts)} texts...")
                    outputs = self.model.embed(texts)
                    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
                    print(f"✅ vLLM processing completed, output shape: {embeddings.shape}")
                    return embeddings
                except Exception as e:
                    print(f"❌ vLLM processing error: {str(e)}")
                    raise

            return await loop.run_in_executor(None, _embed)
        except Exception as e:
            print(f"❌ Asynchronous embedding error: {str(e)}")
            raise
    
    async def similarity_search(self, queries: List[str], documents: List[str], 
                               task_description: str) -> List[List[float]]:
        """Asynchronous similarity search"""
        # Add instructions to queries
        formatted_queries = [self.get_detailed_instruct(task_description, query) for query in queries]
        
        # Merge all texts
        all_texts = formatted_queries + documents
        
        # Get embeddings
        embeddings = await self.embed_texts(all_texts)
        
        # Calculate similarity
        query_embeddings = embeddings[:len(queries)]
        doc_embeddings = embeddings[len(queries):]
        
        scores = (query_embeddings @ doc_embeddings.T)
        return scores.tolist()

# Global service instance and concurrency control
embedding_service = None
processing_lock = threading.Semaphore(16)  # Increase concurrency to 16
vllm_lock = threading.Lock()  # vLLM global lock, ensure only one request at a time
# Batch processing related
batch_queue = []
batch_lock = threading.Lock()
batch_event = threading.Event()
batch_processor_running = False

class BatchRequest:
    def __init__(self, texts: List[str], future: asyncio.Future):
        self.texts = texts
        self.future = future
        self.timestamp = time.time()

def batch_processor():
    """Background batch processor"""
    global batch_processor_running
    batch_processor_running = True

    while batch_processor_running:
        try:
            # Wait for batch or timeout
            batch_event.wait(timeout=0.01)  # 10ms timeout

            with batch_lock:
                if not batch_queue:
                    batch_event.clear()
                    continue

                # Collect current batch
                current_batch = batch_queue.copy()
                batch_queue.clear()
                batch_event.clear()

            if current_batch:
                process_batch(current_batch)

        except Exception as e:
            print(f"❌ Batch processor error: {e}")
            time.sleep(0.1)

def process_batch(batch_requests: List[BatchRequest]):
    """Process a batch of requests"""
    try:
        # Merge all texts
        all_texts = []
        request_indices = []  # Record position of each request's texts in merged list

        for req in batch_requests:
            start_idx = len(all_texts)
            all_texts.extend(req.texts)
            end_idx = len(all_texts)
            request_indices.append((start_idx, end_idx))

        print(f"🔄 Batch processing {len(batch_requests)} requests, total {len(all_texts)} texts")

        # vLLM batch processing
        outputs = embedding_service.model.embed(all_texts)
        embeddings = torch.stack([torch.tensor(o.outputs.embedding) for o in outputs])

        # Distribute results to each request
        for req, (start_idx, end_idx) in zip(batch_requests, request_indices):
            req_embeddings = embeddings[start_idx:end_idx]
            req.future.set_result(req_embeddings)

        print(f"✅ Batch processing completed, processed {len(batch_requests)} requests")
        


    except Exception as e:
        # If batch processing fails, set exception for all requests
        for req in batch_requests:
            if not req.future.done():
                req.future.set_exception(e)
        print(f"❌ Batch processing failed: {e}")

# Create FastAPI application
app = FastAPI(
    title="vLLM Embedding Service API",
    description="Text embedding and similarity search service based on vLLM",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize model when service starts"""
    global embedding_service
    print("🚀 Initializing vLLM embedding service...")
    # Note: Model initialization is completed in main() function, no need to repeat here
    print("✅ Service startup completed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "vllm-embedding"}



@app.post("/embed", response_model=EmbeddingResponse)
async def embed_endpoint(request: EmbeddingRequest):
    """
    Text embedding endpoint - with concurrency control
    """
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Remove application layer concurrency limit, let batch processor handle load balancing
    try:
        start_time = time.time()
        print(f"📝 Received embedding request, text count: {len(request.texts)}")

        # Validate input
        if not request.texts or len(request.texts) == 0:
            raise HTTPException(status_code=400, detail="Text list cannot be empty")

        # Handle instruction formatting
        texts = request.texts
        if request.task_description:
            texts = [embedding_service.get_detailed_instruct(request.task_description, text)
                    for text in texts]
            print(f"🔧 Applied task description: {request.task_description}")

        print(f"🚀 Starting to process {len(texts)} texts...")
        # Use batch processing method
        embeddings = await embedding_service.embed_texts_batch(texts)
        processing_time = time.time() - start_time

        print(f"✅ Embedding completed, time taken: {processing_time:.2f}s, embedding dimension: {embeddings.shape}")

        # Ensure correct data type is returned
        embeddings_list = embeddings.cpu().numpy().tolist() if hasattr(embeddings, 'cpu') else embeddings.tolist()

        return EmbeddingResponse(
            embeddings=embeddings_list,
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Embedding processing error: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding processing failed: {str(e)}")

@app.post("/similarity", response_model=SimilaritySearchResponse)
async def similarity_search_endpoint(request: SimilaritySearchRequest):
    """
    Similarity search endpoint
    """
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        start_time = time.time()
        
        scores = await embedding_service.similarity_search(
            request.queries, 
            request.documents, 
            request.task_description
        )
        
        processing_time = time.time() - start_time
        
        return SimilaritySearchResponse(
            scores=scores,
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='vLLM Embedding API Service')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-Embedding-4B', 
                       help='Model name or path')
    parser.add_argument('--tensor-parallel-size', type=int, default=8,
                       help='Tensor parallel size')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.75,
                       help='GPU memory utilization (cross-node optimization)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Service host address')
    parser.add_argument('--port', type=int, default=2341,
                       help='Service port')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes')

    
    args = parser.parse_args()
    
    # Initialize global service
    global embedding_service
    embedding_service = VLLMEmbeddingAPI(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    # Start batch processor thread
    import threading
    batch_thread = threading.Thread(target=batch_processor, daemon=True)
    batch_thread.start()
    print("🚀 Batch processor started")
    
    print(f"Starting vLLM embedding API service")
    print(f"Service address: http://{args.host}:{args.port}")
    print(f"API documentation: http://{args.host}:{args.port}/docs")
    
    # Start service - for vLLM service, must use single worker mode
    print(f"⚠️  Note: vLLM service enforces single worker mode to avoid GPU memory conflicts")
    print(f"🌐 Network configuration optimized for cross-node access")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,  # Force single worker
        log_level="info",
        access_log=True,
        timeout_keep_alive=600,  # Increase keep-alive to 10 minutes, adapt to cross-node latency
        timeout_graceful_shutdown=60,  # Increase graceful shutdown time
        limit_concurrency=100,  # Significantly increase concurrency limit, support multi-node high concurrency
        limit_max_requests=None,  # Remove request count limit
        backlog=2048,  # Increase connection queue size
        h11_max_incomplete_event_size=16777216,  # Increase HTTP event size limit
    )

if __name__ == "__main__":
    main()
