import unittest
from unittest.mock import patch, MagicMock
from classes.rag_query_processor import RAGQueryProcessor
from classes.llm_client import LLMClient
from classes.chromadb_retriever import ChromaDBRetriever

class TestRAGQueryProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        # Create mock LLMClient
        self.mock_llm_client = MagicMock(spec=LLMClient)
        self.mock_llm_client.query.return_value = "Test LLM response"
        
        # Create mock ChromaDBRetriever
        self.mock_retriever = MagicMock(spec=ChromaDBRetriever)
        
        # Create RAGQueryProcessor instances for different scenarios
        self.processor_with_rag = RAGQueryProcessor(
            llm_client=self.mock_llm_client,
            retriever=self.mock_retriever,
            use_rag=True
        )
        
        self.processor_without_rag = RAGQueryProcessor(
            llm_client=self.mock_llm_client,
            retriever=self.mock_retriever,
            use_rag=False
        )

    def test_initialization(self):
        """Test if RAGQueryProcessor initializes correctly."""
        # Test with RAG enabled
        self.assertTrue(self.processor_with_rag.use_rag)
        self.assertIsNotNone(self.processor_with_rag.retriever)
        self.assertEqual(self.processor_with_rag.llm_client, self.mock_llm_client)
        
        # Test with RAG disabled
        self.assertFalse(self.processor_without_rag.use_rag)
        self.assertIsNone(self.processor_without_rag.retriever)
        self.assertEqual(self.processor_without_rag.llm_client, self.mock_llm_client)

    def test_query_without_rag(self):
        """Test query processing without RAG."""
        query_text = "Test query"
        response = self.processor_without_rag.query(query_text)
        
        # Verify response
        self.assertEqual(response, "Test LLM response")
        
        # Verify LLM was called with correct prompt
        expected_prompt = f"""
        You are an AI assistant answering user queries using retrieved context.
        If the context is insufficient, say 'I don't know'. 

        Context:
        No relevant context found.

        Question:
        {query_text}
        """
        self.mock_llm_client.query.assert_called_once_with(expected_prompt)
        
        # Verify retriever was not called
        self.mock_retriever.query.assert_not_called()

    def test_query_with_rag(self):
        """Test query processing with RAG."""
        query_text = "Test query"
        mock_retrieved_docs = [{
            'context': 'Test context',
            'id': 'test_id',
            'score': 0.8,
            'text': 'Test document text'
        }]
        self.mock_retriever.query.return_value = mock_retrieved_docs
        
        response = self.processor_with_rag.query(query_text)
        
        # Verify response
        self.assertEqual(response, "Test LLM response")
        
        # Verify retriever was called
        self.mock_retriever.query.assert_called_once_with(query_text)
        
        # Verify LLM was called with correct prompt
        expected_prompt = f"""
        You are an AI assistant answering user queries using retrieved context.
        If the context is insufficient, say 'I don't know'. 

        Context:
        Test context

        Question:
        {query_text}
        """
        self.mock_llm_client.query.assert_called_once_with(expected_prompt)

    def test_query_with_rag_no_documents(self):
        """Test query processing with RAG when no documents are found."""
        query_text = "Test query"
        self.mock_retriever.query.return_value = []
        
        response = self.processor_with_rag.query(query_text)
        
        # Verify response
        self.assertEqual(response, "Test LLM response")
        
        # Verify retriever was called
        self.mock_retriever.query.assert_called_once_with(query_text)
        
        # Verify LLM was called with correct prompt
        expected_prompt = f"""
        You are an AI assistant answering user queries using retrieved context.
        If the context is insufficient, say 'I don't know'. 

        Context:
        No relevant context found.

        Question:
        {query_text}
        """
        self.mock_llm_client.query.assert_called_once_with(expected_prompt)

    def test_query_empty_input(self):
        """Test query processing with empty input."""
        response = self.processor_with_rag.query("")
        self.assertIsInstance(response, str)

if __name__ == '__main__':
    unittest.main() 