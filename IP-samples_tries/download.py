import os
import logging
import uuid
import time
import hashlib
import sys
import signal
import openai
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from datetime import datetime
from itertools import islice
from dotenv import load_dotenv
from datasets import load_dataset
from typing import List, Tuple, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from prefect import flow, task, get_run_logger
from prefect.cache_policies import NO_CACHE
from secret_key import AwsSecretManager
from state_manager import StateManager

load_dotenv()

os.environ["PREFECT_API_KEY"] = os.getenv("PREFECT_API_KEY", "")
os.environ["PREFECT_URL"] = os.getenv("PREFECT_URL", "")

secret_key_obj = AwsSecretManager()
is_secret = secret_key_obj.get_secrets()

logging.info(f"Validate is_secret - {is_secret}")

@dataclass
class Config:
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    qdrant_url: str = os.getenv("QDRANT_URL") 
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME")
    dataset_name: str = ""
    chunk_size: int = 512
    overlap: int = 50
    embedding_model: str = ""
    max_retries: int = 5
    base_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    openai_timeout: int = 30
    qdrant_timeout: int = 120
    openai_requests_per_minute: int = 3000
    openai_tokens_per_minute: int = 1000000
    log_level: str = "INFO"
    max_workers: int = 5
    shard_index: int = int(os.getenv("SHARD_INDEX"))
    num_shards: int = 173
    output_dir: str = "downloaded_shards"
    split: str = "train"

class Shutdown:
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        logging.info(f"Received signal {signum}. Initiating shutdown")
        self.shutdown = True

class RateLimiter:
    """Thread-safe rate limiter for API requests and token usage."""
    
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_times = []
        self.token_usage = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self, estimated_tokens: int = 1000):
        """Wait if rate limits would be exceeded."""
        with self.lock:
            now = time.time()
            self.request_times = [t for t in self.request_times if now - t < 60]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if now - t < 60]
            
            if len(self.request_times) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            if current_tokens + estimated_tokens > self.tokens_per_minute:
                sleep_time = 60 - (now - self.token_usage[0][0]) if self.token_usage else 0
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.request_times.append(now)
            self.token_usage.append((now, estimated_tokens))

class DocumentProcessor:
    """Processes a single document through the entire pipeline."""
    
    def __init__(self, config: Config, openai_client: OpenAI, qdrant_client: QdrantClient, rate_limiter: RateLimiter):
        self.config = config
        self.openai_client = openai_client
        self.qdrant_client = qdrant_client
        self.rate_limiter = rate_limiter
    
    def chunking(self, text: str) -> List[str]:
        """Split text into overlapping chunks for processing."""
        if not text or not text.strip():
            return []
        
        words = text.split()
        if len(words) <= self.config.chunk_size:
            return [text.strip()]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + self.config.chunk_size, len(words))
            chunk = " ".join(words[start:end]).strip()
            
            if chunk:
                chunks.append(chunk)
            
            if end == len(words):
                break
                
            start += self.config.chunk_size - self.config.overlap
        
        return chunks
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text chunk."""
        if not text:
            return None
        
        estimated_tokens = len(text.split()) * 1.3
        self.rate_limiter.wait_if_needed(int(estimated_tokens))
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.openai_client.embeddings.create(
                    input=[text],
                    model=self.config.embedding_model
                )
                return response.data[0].embedding
                
            except (openai.RateLimitError, openai.APITimeoutError) as e:
                wait_time = min(
                    self.config.base_retry_delay * (2 ** attempt),
                    self.config.max_retry_delay
                )
                logging.warning(f"API error (attempt {attempt+1}), waiting {wait_time}s: {e}")
                time.sleep(wait_time)
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = min(
                        self.config.base_retry_delay * (2 ** attempt),
                        self.config.max_retry_delay
                    )
                    logging.warning(f"Embedding failed (attempt {attempt+1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All embedding attempts failed: {e}")
                    return None
        
        return None
    
    def upsert_point(self, point: PointStruct) -> bool:
        """Insert single point into Qdrant."""
        for attempt in range(self.config.max_retries):
            try:
                self.qdrant_client.upsert(
                    collection_name=self.config.collection_name,
                    points=[point]
                )
                return True
                
            except (UnexpectedResponse, Exception) as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = min(
                        self.config.base_retry_delay * (2 ** attempt),
                        self.config.max_retry_delay
                    )
                    logging.warning(f"Qdrant upsert failed (attempt {attempt+1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All Qdrant upsert attempts failed: {e}")
                    return False
        
        return False
    
    def create_stable_id(self, record_id: str, chunk_index: int) -> str:
        """Generate deterministic ID for each chunk."""
        content = f"{record_id}::{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_document(self, document: dict, doc_index: int) -> Tuple[int, int]:
        start = time.time()
        text = document.get("text", "").strip()
        if not text:
            return 0, 0

        chunks = self.chunking(text)
        if not chunks:
            return 0, 0

        doc_id = document.get("id", f"doc_{doc_index}")
        chunks_processed = 0
        chunks_failed = 0

        for chunk_idx, chunk in enumerate(chunks):
            try:
                embedding = self.get_embedding(chunk)
                if embedding is None:
                    chunks_failed += 1
                    continue

                stable_id = self.create_stable_id(doc_id, chunk_idx)
                metadata = document.get("metadata", {})
                payload = {
                    "Content": chunk,
                    "ID": stable_id,
                    "Embedding-Model": self.config.embedding_model,
                    "MimeType": "text/plain",
                    "Class Name": "TextNode",
                    "Author": metadata.get("author", ""),
                    "License": metadata.get("license", ""),
                    "URL": metadata.get("url", "")
                }
                point = PointStruct(id=stable_id, vector=embedding, payload=payload)

                if self.upsert_point(point):
                    chunks_processed += 1
                else:
                    chunks_failed += 1

            except Exception as e:
                logging.error(f"[doc_id={doc_id} chunk_idx={chunk_idx}] Error processing chunk: {e}", exc_info=True)
                chunks_failed += 1

        duration_ms = int((time.time() - start) * 1000)
        logging.info(f"[doc_id={doc_id}] processed_chunks={chunks_processed} failed_chunks={chunks_failed} duration_ms={duration_ms}")
        return chunks_processed, chunks_failed

class CaselawIngestionPipeline:
    """Main pipeline for ingesting caselaw data into Qdrant vector database."""
    
    def __init__(self, config: Config):
        self.config = config
        self.shutdown_handler = Shutdown()
        self.rate_limiter = RateLimiter(
            config.openai_requests_per_minute,
            config.openai_tokens_per_minute
        )
        self.setup_logging()
        self.initialize_clients()
        self.state_manager = StateManager(config.shard_index)
        self.state = self.state_manager.state
        logging.info("Pipeline initialized successfully")
        if self.state_manager.is_shard_completed():
            logging.info(f"Shard {config.shard_index} is already completed. Exiting.")
            sys.exit(0)
        
    logging.info("Pipeline initialized successfully")
    
    def setup_logging(self):
        """Configure logging with appropriate format and level."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level))
        console_handler.setFormatter(logging.Formatter(log_format))
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            handlers=[console_handler],
            format=log_format
        )
    
    def initialize_clients(self):
        """Initialize and test OpenAI and Qdrant client connections."""
        try:
            self.openai_client = OpenAI(
                api_key=self.config.openai_api_key,
                timeout=self.config.openai_timeout
            )
            
            self.qdrant_client = QdrantClient(
                url=self.config.qdrant_url,
                timeout=self.config.qdrant_timeout
            )
            
            self.test_connections()
            logging.info("Client connections established successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize clients: {e}")
            raise
    
    @task(retries=3, retry_delay_seconds=5, cache_policy=NO_CACHE)
    def test_connections(self):
        """Verify client connections are working."""
        logger = get_run_logger()
        try:
            self.openai_client.embeddings.create(
                input=["test"],
                model=self.config.embedding_model
            )
            self.qdrant_client.get_collections()
            logger.info("Connection test passed successfully")
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            raise
    
    @task(retries=3, retry_delay_seconds=5, cache_policy=NO_CACHE)
    def setup_collection(self):
        logger = get_run_logger()
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(col.name == self.config.collection_name for col in collections)
            if collection_exists:
                logger.info(f"Collection '{self.config.collection_name}' already exists")
                return
            
            self.qdrant_client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Collection '{self.config.collection_name}' created successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise
    
    @task(retries=3, retry_delay_seconds=10, cache_policy=NO_CACHE)
    def download_and_save_shard(self):

        logger = get_run_logger()
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            shard_filename = f"{self.config.split}_shard_{self.config.shard_index:03d}.jsonl"
            shard_path = os.path.join(self.config.output_dir, shard_filename)
            
            if os.path.exists(shard_path):
                logger.info(f"Shard file {shard_path} already exists, skipping download")
                return shard_path
            
            logger.info(f"Downloading shard {self.config.shard_index} of {self.config.num_shards} from {self.config.dataset_name}")
            
            ds_stream = load_dataset(
                self.config.dataset_name,
                split=self.config.split,
                streaming=True
            )
            
            shard_ds = ds_stream.shard(
                num_shards=self.config.num_shards, 
                index=self.config.shard_index
            )
            
            logger.info(f"Saving shard to {shard_path}")
            with open(shard_path, "w", encoding="utf-8") as f:
                for example in shard_ds:
                    f.write(json.dumps(example) + "\n")
            
            logger.info(f"Successfully saved shard #{self.config.shard_index} to {shard_path}")
            return shard_path
            
        except Exception as e:
            logger.error(f"Failed to download and save shard: {e}")
            raise
    
    @task(cache_policy=NO_CACHE)
    def load_shard_documents(self, shard_path: str):
        """Load documents from the downloaded shard file."""
        logger = get_run_logger()
        try:
            documents = []
            logger.info(f"Loading documents from shard file: {shard_path}")
            
            with open(shard_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if line:
                        try:
                            document = json.loads(line)
                            documents.append(document)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON on line {line_num + 1}: {e}")
                            continue
            
            logger.info(f"Loaded {len(documents)} documents from shard file")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load shard documents: {e}")
            raise
    
    @task(cache_policy=NO_CACHE)
    def process_shard_documents(self, documents: List[dict]):
        """Process all documents from the shard, with resume & checkpointing."""
        logger = get_run_logger()
        try:
            total_docs = len(documents)
            logger.info(f"Shard {self.config.shard_index}: {total_docs} total documents in this shard")

            self.state.documents_in_current_shard = total_docs
            resume_at = self.state.documents_processed_in_shard or 0

            if resume_at >= total_docs:
                logger.info(f"Shard {self.config.shard_index} already fully processed (resume_at={resume_at})")
                return

            if resume_at > 0:
                logger.info(f"Resuming shard {self.config.shard_index} at document {resume_at}/{total_docs}")

            to_process = documents[resume_at:]
            num_to_process = len(to_process)

            local_docs = 0
            local_chunks = 0

            pbar = tqdm(
                desc=f"Processing shard {self.config.shard_index}",
                unit="docs",
                total=num_to_process
            )

            processors = [
                DocumentProcessor(self.config, self.openai_client, self.qdrant_client, self.rate_limiter)
                for _ in range(self.config.max_workers)
            ]

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_idx = {}
                for offset, document in enumerate(to_process):
                    if self.shutdown_handler.shutdown:
                        logger.info("Shutdown requested. Stopping document submission.")
                        break

                    actual_idx = resume_at + offset
                    processor = processors[actual_idx % len(processors)]
                    fut = executor.submit(processor.process_document, document, actual_idx)
                    future_to_idx[fut] = actual_idx

                for fut in as_completed(future_to_idx):
                    if self.shutdown_handler.shutdown:
                        logger.info("Shutdown requested. Breaking out of result loop.")
                        break

                    doc_idx = future_to_idx[fut]
                    try:
                        chunks_done, chunks_failed = fut.result()
                        local_docs += 1
                        local_chunks += chunks_done

                        tqdm.write(
                            f"[shard_processing={self.config.shard_index} doc_processing={doc_idx}] "
                            f"processed_chunks={chunks_done} failed_chunks={chunks_failed}"
                            f"total_chunks={local_chunks}"
                        )
                        pbar.update(1)

                    except Exception:
                        logger.exception(f"[shard={self.config.shard_index} doc={doc_idx}] error processing")
                        local_docs += 1
                        pbar.update(1)

            pbar.close()

            # 5. Update and persist state
            self.state.documents_processed_in_shard += local_docs
            self.state.chunks_processed_in_shard += local_chunks
            self.state.total_documents_processed += local_docs
            self.state.total_chunks_processed += local_chunks

            self.state_manager.state = self.state
            self.state_manager.save_state()

            end_ts = datetime.utcnow().isoformat() + "Z"
            logger.info(
                f"Shard {self.config.shard_index} complete at {end_ts}\n"
                f" - Docs this run: {local_docs}/{num_to_process}\n"
                f" - Chunks this run: {local_chunks}\n"
                f" - Shard docs done: {self.state.documents_processed_in_shard}/{total_docs}\n"
                f" - Cumulative docs: {self.state.total_documents_processed}\n"
                f" - Cumulative chunks: {self.state.total_chunks_processed}"
            )

        except Exception:
            logger.exception("Critical error—saving state before abort")
            self.state_manager.state = self.state
            self.state_manager.save_state()
            raise

    @flow(name=f"Caselaw_Shard_Ingestion_{uuid.uuid4()}", log_prints=True)
    def run(self) -> None:
        """Execute the full pipeline as a Prefect flow."""
        logger = get_run_logger()
        try:
            start_time = time.time()
            logger.info(f"Starting Caselaw Shard Ingestion Pipeline for shard {self.config.shard_index}")
            
            self.setup_collection()
            
            shard_path = self.download_and_save_shard()
            
            documents = self.load_shard_documents(shard_path)
            
            self.process_shard_documents(documents)
            
            total_time = time.time() - start_time
            logger.info(f"Shard {self.config.shard_index} pipeline completed successfully in {total_time:.2f} seconds! Ended at {datetime.utcnow().isoformat()}Z")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    config = Config()
    
    required_vars = ['OPENAI_API_KEY', 'QDRANT_URL']
    missing_vars = [var for var in required_vars if not getattr(config, var.lower())]
    
    if missing_vars:
        logging.error(f"Missing required environment variables: {missing_vars}")
        sys.exit(1)
    
    logging.info(f"Processing shard {config.shard_index} out of {config.num_shards} total shards")
    
    try:
        pipeline = CaselawIngestionPipeline(config)
        pipeline.run()
        logging.info(f"Shard {config.shard_index} pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()