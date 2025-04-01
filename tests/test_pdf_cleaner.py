import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from classes.pdf_cleaner import PDFCleaner

class TestPDFCleaner(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.cleaner = PDFCleaner(chunk_size=100, chunk_overlap=20)
        
    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Test case with various text issues
        test_text = """
        This is a test text with multiple spaces   and
        newlines. It has a URL https://example.com and
        an email test@example.com. It also has page numbers
        1
        2
        3
        and multiple periods...
        """
        
        cleaned = self.cleaner.clean_text(test_text)
        
        # Verify cleaning results
        self.assertNotIn('https://example.com', cleaned)
        self.assertNotIn('test@example.com', cleaned)
        self.assertNotIn('1', cleaned)
        self.assertNotIn('2', cleaned)
        self.assertNotIn('3', cleaned)
        self.assertNotIn('...', cleaned)
        self.assertNotIn('  ', cleaned)  # No double spaces
        
    def test_create_chunks(self):
        """Test text chunking functionality."""
        test_text = "This is a test text. It has multiple sentences. We want to chunk it properly."
        
        chunks = self.cleaner.create_chunks(test_text)
        
        # Verify chunking results
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0)
        
        # Verify chunk overlap
        if len(chunks) > 1:
            overlap = len(chunks[0]) - len(chunks[0].split('.')[-1])
            self.assertGreaterEqual(overlap, self.cleaner.chunk_overlap)
            
        # Verify chunk size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), self.cleaner.chunk_size)
            
    def test_extract_metadata(self):
        """Test metadata extraction functionality."""
        test_pdf_path = Path("test.pdf")
        
        # Mock pdfplumber.open
        with patch('pdfplumber.open') as mock_pdf:
            # Create mock PDF object
            mock_pdf_obj = MagicMock()
            mock_pdf_obj.metadata = {
                'Title': 'Test Title',
                'Author': 'Test Author',
                'Subject': 'Test Subject',
                'Keywords': 'test, pdf'
            }
            mock_pdf_obj.pages = [MagicMock()] * 5  # 5 pages
            mock_pdf.return_value.__enter__.return_value = mock_pdf_obj
            
            metadata = self.cleaner.extract_metadata(test_pdf_path)
            
            # Verify metadata extraction
            self.assertEqual(metadata['title'], 'Test Title')
            self.assertEqual(metadata['author'], 'Test Author')
            self.assertEqual(metadata['subject'], 'Test Subject')
            self.assertEqual(metadata['keywords'], 'test, pdf')
            self.assertEqual(metadata['page_count'], 5)
            self.assertEqual(metadata['file_name'], 'test.pdf')
            
    @patch('pdfplumber.open')
    def test_process_pdf(self, mock_pdf):
        """Test complete PDF processing functionality."""
        test_pdf_path = "test.pdf"
        
        # Mock PDF object
        mock_pdf_obj = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test page content."
        mock_pdf_obj.pages = [mock_page]
        mock_pdf_obj.metadata = {
            'Title': 'Test Title',
            'Author': 'Test Author'
        }
        mock_pdf.return_value.__enter__.return_value = mock_pdf_obj
        
        result = self.cleaner.process_pdf(test_pdf_path)
        
        # Verify processing results
        self.assertIn('chunks', result)
        self.assertIn('metadata', result)
        self.assertIn('full_text', result)
        
        # Verify content
        self.assertIsInstance(result['chunks'], list)
        self.assertIsInstance(result['metadata'], dict)
        self.assertIsInstance(result['full_text'], str)
        
    def test_process_pdf_error_handling(self):
        """Test error handling in PDF processing."""
        with patch('pdfplumber.open') as mock_pdf:
            mock_pdf.side_effect = Exception("Test error")
            
            with self.assertRaises(Exception):
                self.cleaner.process_pdf("nonexistent.pdf")

if __name__ == '__main__':
    unittest.main() 