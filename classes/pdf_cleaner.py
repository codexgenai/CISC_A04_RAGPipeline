import pdfplumber
import re
from pathlib import Path
import logging
from typing import List, Dict, Any
import unicodedata

class PDFCleaner:
    """
    A class to clean and preprocess PDF documents for RAG system.
    Handles text extraction, cleaning, and chunking of PDF documents.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF cleaner with chunking parameters.
        
        Args:
            chunk_size (int): Size of each text chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
        
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing unwanted characters and normalizing whitespace.
        
        Args:
            text (str): Raw text extracted from PDF
            
        Returns:
            str: Cleaned text
        """
        # Remove special characters and normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    def create_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks for better context preservation.
        
        Args:
            text (str): Cleaned text to be chunked
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Find the end of the chunk
            end = start + self.chunk_size
            
            if end >= text_length:
                # Last chunk
                chunks.append(text[start:])
                break
                
            # Find the last period before the chunk end
            last_period = text.rfind('.', start, end)
            if last_period > start:
                end = last_period + 1
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            
        return chunks
    
    def extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from PDF file.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            Dict[str, Any]: PDF metadata
        """
        with pdfplumber.open(pdf_path) as pdf:
            metadata = {
                'title': pdf.metadata.get('Title', ''),
                'author': pdf.metadata.get('Author', ''),
                'subject': pdf.metadata.get('Subject', ''),
                'keywords': pdf.metadata.get('Keywords', ''),
                'page_count': len(pdf.pages),
                'file_name': pdf_path.name
            }
        return metadata
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF file: extract, clean, and chunk text.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict[str, Any]: Processed PDF data including chunks and metadata
        """
        pdf_path = Path(pdf_path)
        self.logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text from PDF
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            
            # Clean the extracted text
            cleaned_text = self.clean_text(text)
            
            # Create chunks
            chunks = self.create_chunks(cleaned_text)
            
            # Extract metadata
            metadata = self.extract_metadata(pdf_path)
            
            self.logger.info(f"Successfully processed PDF: {pdf_path}")
            self.logger.info(f"Created {len(chunks)} chunks")
            
            return {
                'chunks': chunks,
                'metadata': metadata,
                'full_text': cleaned_text
            }
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise 