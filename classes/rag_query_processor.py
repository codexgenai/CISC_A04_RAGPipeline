from .llm_client import LLMClient
from .chromadb_retriever import ChromaDBRetriever
import logging
from typing import List, Dict, Any
# from pathlib import Path
# import json

class RAGQueryProcessor:
    """Processes queries using RAG (Retrieval-Augmented Generation)."""
    
    def __init__(self, llm_client: LLMClient, retriever: ChromaDBRetriever, use_rag: bool = True):
        """Initialize the RAG query processor.
        
        Args:
            llm_client: Client for interacting with the language model
            retriever: Client for retrieving relevant documents
            use_rag: Whether to use RAG (True) or direct LLM query (False)
        """
        self.llm_client = llm_client
        self.retriever = retriever
        self.use_rag = use_rag
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized RAGQueryProcessor: use_rag: {use_rag}")
    
    def _format_prompt(self, context: str, question: str) -> str:
        """Format the prompt using the template from config.
        
        Args:
            context: Retrieved context
            question: User's question
            
        Returns:
            str: Formatted prompt
        """
        if not self.llm_client.prompt_template:
            # Default prompt template if none provided
            return f"""Based on the following context, please answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question:
{question}

Answer:"""
        
        return self.llm_client.prompt_template.format(
            context=context,
            question=question
        )
    
    def query(self, query: str) -> str:
        """Process a query using RAG or direct LLM query.
        
        Args:
            query: The user's query string
            
        Returns:
            str: The model's response
        """
        try:
            if not self.use_rag:
                self.logger.info("Using direct LLM query (RAG disabled)")
                return self.llm_client.query(query)
            
            # Retrieve relevant documents
            self.logger.info("Retrieving relevant documents")
            search_results = self.retriever.query(query, top_k=3)
            
            if not search_results:
                self.logger.warning("No relevant documents found")
                return "I couldn't find any relevant information in the documents to answer your question."
            
            # Combine retrieved contexts
            context = "\n\n".join([
                f"Document {i+1}:\n{result.get('text', '')}"
                for i, result in enumerate(search_results)
            ])
            
            # Format prompt with context and query
            prompt = self._format_prompt(context, query)
            
            # Get response from LLM
            self.logger.info("Generating response with context")
            response = self.llm_client.query(prompt)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise

        # # RAG mode: Retrieve relevant context
        # context = ""
        # if self.use_rag:
        #     self.logger.info("-"*80)
        #     self.logger.info("Using RAG pipeline...")
        #     retrieved_docs = self.retriever.query(query_text)
        #     # context = "\n".join(retrieved_docs[0].get('contexts'))
        #     # print(f"retrieved_docs:\n", retrieved_docs)
        #
        #     if not retrieved_docs:
        #         logging.info("*** No relevant documents found.")
        #     else:
        #         result = retrieved_docs[0]
        #         context = result.get('context', '')
        #
        #         # logging.info(f"Result {idx + 1}:")
        #         logging.info(f"ID: {result.get('id', 'N/A')}")  # Handle missing ID
        #         logging.info(f"Score: {result.get('score', 'N/A')}")
        #         doc_text = result.get('text', '')
        #         preview_text = (doc_text[:150] + "...") if len(doc_text) > 150 else doc_text
        #         logging.info(f"Document: {preview_text}")
        #         logging.info(f"Context: {context}")
        #     self.logger.info("-" * 80)
        #
        # # Construct the final prompt
        # final_prompt = f"Context:\n{context}\n\nUser Query:\n{query_text}" if context else query_text
        # self.logger.debug(f"Prompt to LLM: {final_prompt}")
        # response = self.llm_client.query(final_prompt)
        # self.logger.debug(f"LLM Response: {response}")
        # return response


