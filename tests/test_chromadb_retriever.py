import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from classes.chromadb_retriever import ChromaDBRetriever

class TestChromaDBRetriever(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.collection_name = "test_collection"
        self.vectordb_dir = "test_vectordb"
        self.score_threshold = 0.5
        
        # Create a test instance
        self.retriever = ChromaDBRetriever(
            embedding_model_name=self.embedding_model_name,
            collection_name=self.collection_name,
            vectordb_dir=self.vectordb_dir,
            score_threshold=self.score_threshold
        )

    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_initialization(self, mock_chroma_client, mock_sentence_transformer):
        """Test if ChromaDBRetriever initializes correctly."""
        # Create a new instance with mocked dependencies
        retriever = ChromaDBRetriever(
            embedding_model_name=self.embedding_model_name,
            collection_name=self.collection_name,
            vectordb_dir=self.vectordb_dir,
            score_threshold=self.score_threshold
        )
        
        # Verify initialization
        self.assertEqual(retriever.vectordb_path, Path(self.vectordb_dir))
        self.assertEqual(retriever.score_threshold, self.score_threshold)
        mock_chroma_client.assert_called_once_with(path=str(Path(self.vectordb_dir)))
        mock_sentence_transformer.assert_called_once_with(self.embedding_model_name)

    def test_embed_text(self):
        """Test text embedding generation."""
        test_text = "Test text for embedding"
        embedding = self.retriever.embed_text(test_text)
        
        # Verify embedding is a list of floats
        self.assertIsInstance(embedding, list)
        self.assertTrue(all(isinstance(x, float) for x in embedding))

    def test_extract_context(self):
        """Test context extraction from text."""
        test_text = "First paragraph.\n\nSecond paragraph with search term.\n\nThird paragraph."
        search_term = "search term"
        
        # Test with matching paragraph
        context = self.retriever.extract_context(test_text, search_term)
        self.assertEqual(context, "Second paragraph with search term.")
        
        # Test with no matching paragraph
        context = self.retriever.extract_context(test_text, "nonexistent")
        self.assertEqual(context, test_text[:300])

    @patch('sentence_transformers.SentenceTransformer')
    @patch('chromadb.PersistentClient')
    def test_query(self, mock_chroma_client, mock_sentence_transformer):
        """Test document retrieval query."""
        # Mock the collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "metadatas": [[{"text": "Test document 1", "source": "source1"},
                          {"text": "Test document 2", "source": "source2"}]],
            "distances": [[0.3, 0.6]]
        }
        
        # Mock the client
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client
        
        # Mock the embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_model
        
        # Create retriever and test query
        retriever = ChromaDBRetriever(
            embedding_model_name=self.embedding_model_name,
            collection_name=self.collection_name,
            vectordb_dir=self.vectordb_dir,
            score_threshold=self.score_threshold
        )
        
        results = retriever.query("test query")
        
        # Verify results
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)
        self.assertIn("id", results[0])
        self.assertIn("score", results[0])
        self.assertIn("context", results[0])
        self.assertIn("source", results[0])
        
        # Verify that low-score results are filtered
        self.assertTrue(all(result["score"] < self.score_threshold for result in results))

    def test_query_empty_input(self):
        """Test query with empty input."""
        results = self.retriever.query("")
        self.assertIsInstance(results, list)

if __name__ == '__main__':
    unittest.main() 