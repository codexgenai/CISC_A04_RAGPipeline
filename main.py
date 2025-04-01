import os
# import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from classes.config_manager import ConfigManager
from classes.document_ingestor import DocumentIngestor
from classes.embedding_preparer import EmbeddingPreparer
from classes.embedding_loader import EmbeddingLoader
from classes.llm_client import LLMClient
from classes.chromadb_retriever import ChromaDBRetriever
from classes.rag_query_processor import RAGQueryProcessor
from classes.pdf_cleaner import PDFCleaner
from classes.utilities import delete_directory

class PerformanceMonitor:
    """Monitors and logs system performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            'response_times': [],
            'retrieval_times': [],
            'processing_times': [],
            'error_count': 0
        }
        self.logger = logging.getLogger(__name__)
    
    def start_timer(self, operation: str) -> float:
        """Start timing an operation."""
        return time.time()
    
    def end_timer(self, start_time: float, operation: str):
        """End timing an operation and record the duration."""
        duration = time.time() - start_time
        if operation == 'response':
            self.metrics['response_times'].append(duration)
        elif operation == 'retrieval':
            self.metrics['retrieval_times'].append(duration)
        elif operation == 'processing':
            self.metrics['processing_times'].append(duration)
    
    def record_error(self):
        """Record an error occurrence."""
        self.metrics['error_count'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'avg_response_time': sum(self.metrics['response_times']) / len(self.metrics['response_times']) if self.metrics['response_times'] else 0,
            'avg_retrieval_time': sum(self.metrics['retrieval_times']) / len(self.metrics['retrieval_times']) if self.metrics['retrieval_times'] else 0,
            'avg_processing_time': sum(self.metrics['processing_times']) / len(self.metrics['processing_times']) if self.metrics['processing_times'] else 0,
            'total_errors': self.metrics['error_count']
        }

def setup_logging(log_level):
    """ Configures logging to console and file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    numeric_level = getattr(logging, log_level.upper(), logging.DEBUG)
    log_filename = f"rag_pipeline_{datetime.now().strftime('%Y%m%d_%H%M_%S%f')[:-4]}"  # Remove last 4 digits of microseconds
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] %(levelname)s %(module)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / log_filename)
        ]
    )
    logging.getLogger("transformers").setLevel(logging.INFO)
    logging.getLogger("pdfplumber").setLevel(logging.INFO)
    logging.getLogger("chromadb").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)  # Reduce excessive API logs


def ensure_directories_exist(config: ConfigManager):
    """Ensures all required directories exist."""
    for dir_name in config.get_directory_names():
        dir_path = Path(config.get(dir_name))
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured directory exists: {dir_path}")


def step01_ingest_documents(args, monitor: PerformanceMonitor, config: ConfigManager):
    """ Step 01: Reads and preprocesses documents."""
    logging.info("[Step 01] Document ingestion started.")
    start_time = monitor.start_timer('processing')

    try:
        file_list = [args.input_filename] if args.input_filename != "all" else os.listdir(config.get("raw_input_directory"))
        ingestor = DocumentIngestor(file_list=file_list,
                                    input_dir=config.get("raw_input_directory"),
                                    output_dir=config.get("cleaned_text_directory"),
                                    embedding_model_name=config.get("embedding_model_name"))
        ingestor.process_files()
        
        monitor.end_timer(start_time, 'processing')
        logging.info("[Step 01] Document ingestion completed.")
    except Exception as e:
        monitor.record_error()
        logging.error(f"Error in document ingestion: {str(e)}")
        raise


def step02_generate_embeddings(args, monitor: PerformanceMonitor, config: ConfigManager):
    """ Step 02: Generates vector embeddings from text chunks."""
    logging.info("[Step 02] Embedding generation started.")
    start_time = monitor.start_timer('processing')

    try:
        file_list = [args.input_filename] if args.input_filename != "all" else os.listdir(config.get("cleaned_text_directory"))
        preparer = EmbeddingPreparer(file_list=file_list,
                                     input_dir=config.get("cleaned_text_directory"),
                                     output_dir=config.get("embeddings_directory"),
                                     embedding_model_name=config.get("embedding_model_name"))
        preparer.process_files()
        
        monitor.end_timer(start_time, 'processing')
        logging.info("[Step 02] Embedding generation completed.")
    except Exception as e:
        monitor.record_error()
        logging.error(f"Error in embedding generation: {str(e)}")
        raise


def step03_store_vectors(args, monitor: PerformanceMonitor, config: ConfigManager):
    """ Step 03: Stores embeddings in a vector database."""
    logging.info("[Step 03] Vector storage started.")
    start_time = monitor.start_timer('processing')

    try:
        # delete existing vectordb
        if Path(config.get("vectordb_directory")).exists():
            logging.info("deleting existing vectordb")
            delete_directory(config.get("vectordb_directory"))

        file_list = [args.input_filename] if args.input_filename != "all" else os.listdir(config.get("cleaned_text_directory"))
        loader = EmbeddingLoader(cleaned_text_file_list=file_list,
                                 cleaned_text_dir=config.get("cleaned_text_directory"),
                                 embeddings_dir=config.get("embeddings_directory"),
                                 vectordb_dir=config.get("vectordb_directory"),
                                 collection_name=config.get("collection_name"))
        loader.process_files()
        
        monitor.end_timer(start_time, 'processing')
        logging.info("[Step 03] Vector storage completed.")
    except Exception as e:
        monitor.record_error()
        logging.error(f"Error in vector storage: {str(e)}")
        raise


def step04_retrieve_relevant_chunks(args, monitor: PerformanceMonitor, config: ConfigManager):
    """ Step 04: Retrieves relevant text chunks based on a query."""
    logging.info("[Step 04] Retrieval started.")
    start_time = monitor.start_timer('retrieval')

    try:
        logging.info( f"Query arguments: {args.query_args}")

        retriever = ChromaDBRetriever(vectordb_dir=config.get("vectordb_directory"),
                                     embedding_model_name=config.get("embedding_model_name"),
                                     collection_name=config.get("collection_name"),
                                     score_threshold=float(config.get("retriever_min_score_threshold")))

        search_results = retriever.query(args.query_args, top_k=3)
        monitor.end_timer(start_time, 'retrieval')

        if not search_results:
            logging.info("*** No relevant documents found.")
        else:
            for idx, result in enumerate(search_results):
                logging.info(f"Result {idx + 1}:")
                doc_text = result.get('text', '')
                preview_text = (doc_text[:150] + "...") if len(doc_text) > 250 else doc_text

                logging.info(f"ID: {result.get('id', 'N/A')}")
                logging.info(f"Score: {result.get('score', 'N/A')}")
                logging.info(f"Document: {preview_text}")
                logging.info(f"Context: {result.get('context', '')}")
                logging.info("-" * 50)

        logging.info("[Step 04] Retrieval completed.")
    except Exception as e:
        monitor.record_error()
        logging.error(f"Error in retrieval: {str(e)}")
        raise


def step05_generate_response(args, monitor: PerformanceMonitor, config: ConfigManager):
    """ Step 05: Uses LLM to generate an augmented response."""
    logging.info("[Step 05] Response generation started.")
    start_time = monitor.start_timer('response')

    try:
        # Get prompt template from config
        prompt_template = config.get("prompt_template")
        if not prompt_template:
            raise ValueError("prompt_template not found in config file")

        llm_client = LLMClient(llm_api_url=config.get("llm_api_url"),
                              llm_model_name=config.get("llm_model_name"),
                              model_type=args.model_type,
                              prompt_template=prompt_template)

        retriever = ChromaDBRetriever(vectordb_dir=config.get("vectordb_directory"),
                                     embedding_model_name=config.get("embedding_model_name"),
                                     collection_name=config.get("collection_name"))

        processor = RAGQueryProcessor(llm_client=llm_client,
                                    retriever=retriever,
                                    use_rag=args.use_rag)
        response = processor.query(args.query_args)
        monitor.end_timer(start_time, 'response')
        print("\nResponse:\n", response)

        logging.info("[Step 05] Response generation completed.")
    except Exception as e:
        monitor.record_error()
        logging.error(f"Error in response generation: {str(e)}")
        raise


def main():
    print("rag_pipeline starting...")
    """ Command-line interface for the RAG pipeline."""
    parser = argparse.ArgumentParser(description="CLI for RAG pipeline.")
    parser.add_argument("step",
                        choices=["step01_ingest",
                                 "step02_generate_embeddings",
                                 "step03_store_vectors",
                                 "step04_retrieve_chunks",
                                 "step05_generate_response"],
                        help="Specify the pipeline step.")

    parser.add_argument("--input_filename",
                        nargs="?",
                        default=None,
                        help="Specify filename or 'all' to process all files in the input directory. (Optional)")

    parser.add_argument("--query_args",
                        nargs="?",
                        default=None,
                        help="Specify search query arguments (enclosed in quotes) for step 4. (Optional, required for step04_retrieve_chunks)")

    parser.add_argument("--use_rag",
                        action="store_true",
                        help="Call vectordb for RAG before sending to LLM")

    parser.add_argument("--model_type",
                       choices=["causal", "seq2seq"],
                       default="causal",
                       help="Type of model to use - causal for OPT or seq2seq for FLAN-T5")

    args = parser.parse_args()

    # Select config file based on model type
    CONFIG_FILE = f"config_{args.model_type}.json"
    try:
        config = ConfigManager(CONFIG_FILE)
    except FileNotFoundError:
        print(f"Error: Config file {CONFIG_FILE} not found. Please ensure it exists.")
        return

    # Ensure that query_args is required only when using step04_retrieve_chunks
    if args.step in ["step04_retrieve_chunks", "step05_generate_response"] and args.query_args is None:
        parser.error("The 'query_args' parameter is required when using step04_retrieve_chunks or step05_generate_response.")

    if args.step == "step05_generate_response" and args.use_rag is None:
        parser.error("The 'use_rag' parameter is required when using step05_generate_response.")

    setup_logging(config.get("log_level", "DEBUG"))

    logging.info("------ Command line arguments -------")
    logging.info(f"{'step':<50}: {args.step}")
    logging.info(f"{'input_filename':<50}: {args.input_filename}")
    logging.info(f"{'query_args':<50}: {args.query_args}")
    logging.info(f"{'use_rag':<50}: {args.use_rag}")
    logging.info(f"{'model_type':<50}: {args.model_type}")
    logging.info(f"{'config_file':<50}: {CONFIG_FILE}")
    logging.info("------ Config Settings -------")
    for key in sorted(config.to_dict().keys()):
        logging.info(f"{key:<50}: {config.get(key)}")
    logging.info("------------------------------")
    ensure_directories_exist(config)

    # Initialize performance monitor
    monitor = PerformanceMonitor()

    steps = {
        "step01_ingest": step01_ingest_documents,
        "step02_generate_embeddings": step02_generate_embeddings,
        "step03_store_vectors": step03_store_vectors,
        "step04_retrieve_chunks": step04_retrieve_relevant_chunks,
        "step05_generate_response": step05_generate_response
    }

    try:
        steps[args.step](args, monitor, config)
        # Log performance metrics
        metrics = monitor.get_metrics()
        logging.info("------ Performance Metrics -------")
        logging.info(f"Average Response Time: {metrics['avg_response_time']:.2f}s")
        logging.info(f"Average Retrieval Time: {metrics['avg_retrieval_time']:.2f}s")
        logging.info(f"Average Processing Time: {metrics['avg_processing_time']:.2f}s")
        logging.info(f"Total Errors: {metrics['total_errors']}")
        logging.info("--------------------------------")
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

    logging.info(f"RAG pipeline done")

def check_things():
    print("checking things")
    # Use default causal model config for checking
    config = ConfigManager("config_causal.json")
    print("1. creating ChromaDBRetriever")
    retriever = ChromaDBRetriever(vectordb_dir=config.get("vectordb_directory"),
                                  embedding_model_name=config.get("embedding_model_name"),
                                  collection_name=config.get("collection_name"))

    print(f"Collection count: {retriever.collection.count()}")

if __name__ == "__main__":
    # print hi
    # check_things()
    main()
