import unittest
from unittest.mock import patch, MagicMock
import torch
from classes.llm_client import LLMClient

class TestLLMClient(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.model_name = "facebook/opt-350m"
        self.api_url = ""
        self.llm_client = LLMClient(self.api_url, self.model_name)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_initialization(self, mock_model, mock_tokenizer):
        """Test if LLMClient initializes correctly."""
        # Mock the tokenizer and model
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        # Create a new instance
        client = LLMClient(self.api_url, self.model_name)
        
        # Verify that the model and tokenizer were initialized
        mock_tokenizer.assert_called_once_with(self.model_name)
        mock_model.assert_called_once_with(self.model_name)
        
        # Verify attributes are set correctly
        self.assertEqual(client.llm_api_url, self.api_url)
        self.assertEqual(client.llm_model_name, self.model_name)

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_query_success(self, mock_model, mock_tokenizer):
        """Test successful query generation."""
        # Mock the tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer_instance.decode.return_value = "Test response"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock the model
        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4]])
        mock_model.return_value = mock_model_instance
        
        # Create client and test query
        client = LLMClient(self.api_url, self.model_name)
        response = client.query("Test prompt")
        
        # Verify the response
        self.assertEqual(response, "Test response")
        
        # Verify that the tokenizer and model were called correctly
        mock_tokenizer_instance.encode.assert_called_once()
        mock_model_instance.generate.assert_called_once()

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_query_error_handling(self, mock_model, mock_tokenizer):
        """Test error handling during query generation."""
        # Mock the tokenizer to raise an exception
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.encode.side_effect = Exception("Test error")
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Create client and test query
        client = LLMClient(self.api_url, self.model_name)
        response = client.query("Test prompt")
        
        # Verify error handling
        self.assertEqual(response, "Error: Could not generate response from the model.")

    def test_query_empty_prompt(self):
        """Test query with empty prompt."""
        response = self.llm_client.query("")
        self.assertIsInstance(response, str)

if __name__ == '__main__':
    unittest.main() 