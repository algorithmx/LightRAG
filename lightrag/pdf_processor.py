"""
PDF Processing Wrapper for LightRAG

This module provides a flexible PDF processing interface that allows users to
customize PDF text extraction while providing sensible defaults.

Usage Examples:

1. Basic usage (automatic processor selection):
    from lightrag.pdf_processor import extract_pdf_text
    text = extract_pdf_text("document.pdf")

2. Custom processor:
    from lightrag.pdf_processor import BasePDFProcessor, set_pdf_processor

    class MyPDFProcessor(BasePDFProcessor):
        def extract_text(self, pdf_path):
            # Your custom implementation
            return "extracted text"

        def is_available(self):
            return True

    set_pdf_processor(MyPDFProcessor())

3. In transfer module (automatic integration):
    The transfer module automatically uses the configured PDF processor
    for processing PDF documents during knowledge graph transfer.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union
import warnings

logger = logging.getLogger(__name__)


class BasePDFProcessor(ABC):
    """
    Abstract base class for PDF processors.
    
    This allows users to implement custom PDF processing logic
    while maintaining a consistent interface.
    """
    
    @abstractmethod
    def extract_text(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            Exception: If text extraction fails
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the PDF processor is available/installed.
        
        Returns:
            True if processor is available, False otherwise
        """
        pass
    
    def get_metadata(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract metadata from PDF (optional implementation).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        return {}


class PyPDF2Processor(BasePDFProcessor):
    """Default PDF processor using PyPDF2 library."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def is_available(self) -> bool:
        """Check if PyPDF2 is available."""
        try:
            import PyPDF2
            return True
        except ImportError:
            return False
    
    def extract_text(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text using PyPDF2.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                if len(pdf_reader.pages) == 0:
                    self.logger.warning(f"PDF has no pages: {pdf_path}")
                    return ""
                
                text_content = []
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(page_text)
                    except Exception as e:
                        self.logger.warning(f"Failed to extract text from page {page_num} in {pdf_path}: {str(e)}")
                        continue
                
                return "\n".join(text_content)
                
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF {pdf_path}: {str(e)}")
    
    def get_metadata(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract basic metadata from PDF."""
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                metadata = {
                    "page_count": len(pdf_reader.pages),
                    "encrypted": pdf_reader.is_encrypted
                }
                
                # Try to get document info
                if pdf_reader.metadata:
                    doc_info = pdf_reader.metadata
                    metadata.update({
                        "title": doc_info.get("/Title", ""),
                        "author": doc_info.get("/Author", ""),
                        "subject": doc_info.get("/Subject", ""),
                        "creator": doc_info.get("/Creator", ""),
                        "producer": doc_info.get("/Producer", ""),
                        "creation_date": str(doc_info.get("/CreationDate", "")),
                        "modification_date": str(doc_info.get("/ModDate", ""))
                    })
                
                return metadata
                
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata from {pdf_path}: {str(e)}")
            return {"error": str(e)}


class PyMuPDFProcessor(BasePDFProcessor):
    """Alternative PDF processor using PyMuPDF (fitz) library."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def is_available(self) -> bool:
        """Check if PyMuPDF is available."""
        try:
            import fitz  # PyMuPDF
            return True
        except ImportError:
            return False
    
    def extract_text(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install PyMuPDF")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(str(pdf_path))
            text_content = []
            
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    if page_text.strip():
                        text_content.append(page_text)
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from page {page_num} in {pdf_path}: {str(e)}")
                    continue
            
            doc.close()
            return "\n".join(text_content)
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF {pdf_path}: {str(e)}")
    
    def get_metadata(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata using PyMuPDF."""
        try:
            import fitz
            
            doc = fitz.open(str(pdf_path))
            metadata = doc.metadata
            metadata["page_count"] = len(doc)
            doc.close()
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata from {pdf_path}: {str(e)}")
            return {"error": str(e)}


class PDFProcessorManager:
    """
    Manager class for PDF processing with fallback support.
    
    This class provides a unified interface for PDF processing with
    automatic fallback between different processors.
    """
    
    def __init__(self, custom_processor: Optional[BasePDFProcessor] = None):
        """
        Initialize PDF processor manager.
        
        Args:
            custom_processor: Custom PDF processor implementation.
                            If None, will use default processors with fallback.
        """
        self.logger = logging.getLogger(__name__)
        self.custom_processor = custom_processor
        
        # Default processors in order of preference
        self.default_processors = [
            PyMuPDFProcessor(),  # Generally better text extraction
            PyPDF2Processor(),   # Fallback option
        ]
    
    def get_available_processor(self) -> Optional[BasePDFProcessor]:
        """
        Get the first available PDF processor.
        
        Returns:
            Available PDF processor or None if none available
        """
        # Use custom processor if provided and available
        if self.custom_processor and self.custom_processor.is_available():
            return self.custom_processor
        
        # Try default processors
        for processor in self.default_processors:
            if processor.is_available():
                return processor
        
        return None
    
    def extract_text(self, pdf_path: Union[str, Path]) -> str:
        """
        Extract text from PDF using available processor.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            Exception: If no PDF processor is available or extraction fails
        """
        processor = self.get_available_processor()
        
        if processor is None:
            raise Exception(
                "No PDF processor available. Please install PyPDF2 or PyMuPDF:\n"
                "  pip install PyPDF2\n"
                "  pip install PyMuPDF"
            )
        
        self.logger.debug(f"Using {processor.__class__.__name__} for PDF processing")
        return processor.extract_text(pdf_path)
    
    def get_metadata(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract metadata from PDF using available processor.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        processor = self.get_available_processor()
        
        if processor is None:
            return {"error": "No PDF processor available"}
        
        return processor.get_metadata(pdf_path)
    
    def is_pdf_processing_available(self) -> bool:
        """
        Check if PDF processing is available.
        
        Returns:
            True if at least one PDF processor is available
        """
        return self.get_available_processor() is not None


# Global PDF processor manager instance
_pdf_manager = PDFProcessorManager()


def set_pdf_processor(processor: BasePDFProcessor) -> None:
    """
    Set a custom PDF processor globally.
    
    Args:
        processor: Custom PDF processor implementation
        
    Example:
        # Create custom processor
        class MyPDFProcessor(BasePDFProcessor):
            def extract_text(self, pdf_path):
                # Custom implementation
                return "extracted text"
            
            def is_available(self):
                return True
        
        # Set as global processor
        set_pdf_processor(MyPDFProcessor())
    """
    global _pdf_manager
    _pdf_manager = PDFProcessorManager(custom_processor=processor)


def extract_pdf_text(pdf_path: Union[str, Path]) -> str:
    """
    Extract text from PDF using the global PDF processor.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    return _pdf_manager.extract_text(pdf_path)


def get_pdf_metadata(pdf_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract metadata from PDF using the global PDF processor.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing PDF metadata
    """
    return _pdf_manager.get_metadata(pdf_path)


def is_pdf_processing_available() -> bool:
    """
    Check if PDF processing is available globally.
    
    Returns:
        True if PDF processing is available
    """
    return _pdf_manager.is_pdf_processing_available()
