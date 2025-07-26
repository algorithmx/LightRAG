"""
LightRAG PDF Processing Example

This example demonstrates how to use LightRAG's PDF processing capabilities
across different parts of the system: direct insertion, API usage, and transfer.
"""

import os
import asyncio
from pathlib import Path

from lightrag import (
    LightRAG, 
    BasePDFProcessor, 
    set_pdf_processor, 
    extract_pdf_text,
    is_pdf_processing_available
)
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc


# ============================================================================
# Custom PDF Processor Example
# ============================================================================

class EnhancedPDFProcessor(BasePDFProcessor):
    """
    Example of a custom PDF processor with enhanced text extraction.
    
    This processor adds preprocessing and postprocessing to improve
    text quality from PDF extraction.
    """
    
    def __init__(self):
        self.logger = None
        
    def is_available(self) -> bool:
        """Check if PyMuPDF is available (preferred for this processor)"""
        try:
            import fitz  # PyMuPDF
            return True
        except ImportError:
            try:
                import PyPDF2
                return True
            except ImportError:
                return False
    
    def extract_text(self, pdf_path) -> str:
        """Enhanced PDF text extraction with preprocessing"""
        import re
        
        # Try PyMuPDF first (better text extraction)
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if page_text.strip():
                    # Enhanced preprocessing
                    cleaned_text = self._preprocess_text(page_text)
                    if cleaned_text:
                        text_content.append(cleaned_text)
            
            doc.close()
            final_text = "\n\n".join(text_content)
            
        except ImportError:
            # Fallback to PyPDF2
            import PyPDF2
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text_content = []
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text.strip():
                        cleaned_text = self._preprocess_text(page_text)
                        if cleaned_text:
                            text_content.append(cleaned_text)
                
                final_text = "\n\n".join(text_content)
        
        return self._postprocess_text(final_text)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess extracted text to improve quality"""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)     # Add space after periods
        
        # Remove page numbers and headers/footers (basic heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip likely page numbers
            if re.match(r'^\d+$', line):
                continue
            # Skip very short lines that might be headers/footers
            if len(line) < 10 and not re.search(r'[.!?]$', line):
                continue
            if line:
                cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    def _postprocess_text(self, text: str) -> str:
        """Final postprocessing of the complete text"""
        import re
        
        # Fix sentence boundaries
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


# ============================================================================
# Basic PDF Processing Examples
# ============================================================================

def basic_pdf_processing_example():
    """Demonstrate basic PDF processing capabilities"""
    print("=== Basic PDF Processing Example ===")
    
    # Check if PDF processing is available
    if is_pdf_processing_available():
        print("✓ PDF processing is available")
        
        # Example: Extract text from a PDF file
        pdf_path = "sample_document.pdf"  # Replace with actual PDF path
        
        if Path(pdf_path).exists():
            try:
                text = extract_pdf_text(pdf_path)
                print(f"Extracted {len(text)} characters from PDF")
                print(f"Preview: {text[:200]}...")
            except Exception as e:
                print(f"Failed to extract text: {str(e)}")
        else:
            print(f"Sample PDF not found: {pdf_path}")
    else:
        print("✗ PDF processing not available. Install PyPDF2 or PyMuPDF:")
        print("  pip install PyPDF2")
        print("  pip install PyMuPDF")


def custom_pdf_processor_example():
    """Demonstrate custom PDF processor setup"""
    print("\n=== Custom PDF Processor Example ===")
    
    # Set up custom PDF processor
    custom_processor = EnhancedPDFProcessor()
    
    if custom_processor.is_available():
        set_pdf_processor(custom_processor)
        print("✓ Custom PDF processor set successfully")
        
        # Test the custom processor
        pdf_path = "sample_document.pdf"
        if Path(pdf_path).exists():
            try:
                text = extract_pdf_text(pdf_path)
                print(f"Custom processor extracted {len(text)} characters")
                print(f"Preview: {text[:200]}...")
            except Exception as e:
                print(f"Custom processor failed: {str(e)}")
        else:
            print(f"Sample PDF not found: {pdf_path}")
    else:
        print("✗ Custom PDF processor dependencies not available")


# ============================================================================
# LightRAG Integration Examples
# ============================================================================

async def lightrag_pdf_integration_example():
    """Demonstrate PDF processing with LightRAG"""
    print("\n=== LightRAG PDF Integration Example ===")
    
    # Setup LightRAG instance
    async def openai_embedding_func(texts):
        return await openai_embed(texts=texts, model="text-embedding-3-small")
    
    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=openai_embedding_func
    )
    
    rag = LightRAG(
        working_dir="./rag_storage_pdf_example",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=embedding_func
    )
    
    # Example 1: Direct file insertion with PDF support
    pdf_files = ["document1.pdf", "document2.pdf", "research_paper.pdf"]
    
    for pdf_file in pdf_files:
        if Path(pdf_file).exists():
            print(f"Processing PDF: {pdf_file}")
            success = rag.insert_file(pdf_file)
            if success:
                print(f"✓ Successfully inserted {pdf_file}")
            else:
                print(f"✗ Failed to insert {pdf_file}")
        else:
            print(f"PDF not found: {pdf_file}")
    
    # Example 2: Query the knowledge base
    if any(Path(pdf).exists() for pdf in pdf_files):
        print("\nQuerying the knowledge base...")
        
        test_queries = [
            "What are the main topics discussed in the documents?",
            "Summarize the key findings from the research papers.",
            "What methodologies are mentioned in the documents?"
        ]
        
        for query in test_queries:
            try:
                result = rag.query(query)
                print(f"\nQuery: {query}")
                print(f"Answer: {result[:300]}...")
            except Exception as e:
                print(f"Query failed: {str(e)}")


# ============================================================================
# Transfer Module Integration
# ============================================================================

async def pdf_transfer_example():
    """Demonstrate PDF processing in knowledge graph transfer"""
    print("\n=== PDF Transfer Example ===")
    
    from lightrag.transfer import KnowledgeGraphTransfer, TransferConfig
    
    # Setup source RAG with PDFs
    source_rag = LightRAG(working_dir="./rag_storage_source_pdf")
    
    # Create transfer instance
    transfer = KnowledgeGraphTransfer(source_rag)
    
    # Export with PDF documents
    config = TransferConfig(
        include_documents=True,
        include_embeddings=False,
        regenerate_embeddings=True
    )
    
    # Specify PDF documents to include in transfer
    pdf_documents = [
        "./documents/research_papers/",
        "./documents/manuals/user_guide.pdf"
    ]
    
    try:
        package_path = transfer.export_for_transfer(
            output_path="./pdf_transfer_package",
            config=config,
            document_paths=pdf_documents
        )
        print(f"✓ Transfer package created: {package_path}")
        
        # The transfer module will automatically handle PDF processing
        # when importing to the target system
        
    except Exception as e:
        print(f"Transfer failed: {str(e)}")


# ============================================================================
# API Integration Example
# ============================================================================

def api_pdf_support_info():
    """Show information about PDF support in the API"""
    print("\n=== API PDF Support Information ===")
    
    print("PDF files are automatically supported in the LightRAG API when:")
    print("1. PDF processing libraries are installed (PyPDF2 or PyMuPDF)")
    print("2. Files are uploaded through the /documents/upload endpoint")
    print("3. Files are scanned from the input directory")
    
    print("\nAPI endpoints that support PDF:")
    print("- POST /documents/upload - Upload PDF files directly")
    print("- POST /documents/scan - Scan directory for PDF files")
    print("- The DocumentManager automatically detects PDF support")
    
    if is_pdf_processing_available():
        print("\n✓ PDF processing is currently available for API use")
    else:
        print("\n✗ PDF processing not available - install dependencies")


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Run all PDF processing examples"""
    print("LightRAG PDF Processing Examples")
    print("=" * 50)
    
    # Set environment variables (customize as needed)
    os.environ.setdefault("OPENAI_API_KEY", "your-openai-api-key")
    
    # Create example directories
    os.makedirs("./rag_storage_pdf_example", exist_ok=True)
    os.makedirs("./rag_storage_source_pdf", exist_ok=True)
    os.makedirs("./documents/research_papers", exist_ok=True)
    
    try:
        # Run examples
        basic_pdf_processing_example()
        custom_pdf_processor_example()
        await lightrag_pdf_integration_example()
        await pdf_transfer_example()
        api_pdf_support_info()
        
        print("\n" + "=" * 50)
        print("PDF Processing Examples Completed!")
        print("\nNext steps:")
        print("1. Add your PDF files to test with real documents")
        print("2. Customize the PDF processor for your specific needs")
        print("3. Integrate PDF processing into your application workflow")
        
    except Exception as e:
        print(f"\nExample execution failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
