"""
LightRAG Knowledge Graph Transfer Example

This example demonstrates how to transfer a knowledge graph from a high-performance
server (Computer A) to a local query server (Computer B) with different LLM and
embedding models.

Scenario:
- Computer A: High-performance external LLM (GPT-4) + OpenAI embeddings
- Computer B: Local LLM (Llama) + Local embeddings (Ollama)
"""

import os
import asyncio
from pathlib import Path

from lightrag import LightRAG, KnowledgeGraphTransfer
from lightrag.transfer import TransferConfig, ValidationConfig
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.llm.ollama import ollama_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.pdf_processor import BasePDFProcessor, set_pdf_processor


# ============================================================================
# COMPUTER A: Export Knowledge Graph (High-performance setup)
# ============================================================================

async def setup_source_rag():
    """Setup source RAG instance with high-performance models"""
    
    # Configure OpenAI embedding function
    async def openai_embedding_func(texts):
        return await openai_embed(
            texts=texts,
            model="text-embedding-3-large"
        )
    
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=openai_embedding_func
    )
    
    # Create source RAG instance
    source_rag = LightRAG(
        working_dir="./rag_storage_source",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=embedding_func,
        # Optimized for high-performance processing
        chunk_token_size=1200,
        entity_extract_max_gleaning=2,
        max_async=16
    )
    
    return source_rag


async def export_knowledge_graph():
    """Export knowledge graph from Computer A"""
    print("=== COMPUTER A: Exporting Knowledge Graph ===")
    
    # Setup source RAG
    source_rag = await setup_source_rag()
    
    # Create transfer instance
    transfer = KnowledgeGraphTransfer(source_rag)
    
    # Configure export - optimized for different embedding models
    export_config = TransferConfig(
        include_embeddings=False,  # Skip embeddings - target uses different model
        include_llm_cache=True,    # Include cache to save LLM processing time
        include_documents=True,    # Include documents for re-processing
        compression_enabled=True,  # Compress for efficient transfer
        verify_integrity=True      # Verify export integrity
    )
    
    # Specify document paths to include
    document_paths = [
        "./documents/folder_X",  # Main document folder
        "./additional_docs"      # Any additional documents
    ]
    
    # Export knowledge graph
    print("Exporting knowledge graph...")
    package_path = transfer.export_for_transfer(
        output_path="./transfer_package",
        config=export_config,
        document_paths=document_paths
    )
    
    print(f"Export completed: {package_path}")
    print(f"Package size: {Path(package_path).stat().st_size / (1024*1024):.2f} MB")
    
    return package_path


# ============================================================================
# COMPUTER B: Import Knowledge Graph (Local setup)
# ============================================================================

async def setup_target_rag():
    """Setup target RAG instance with local models"""
    
    # Configure local embedding function
    async def local_embedding_func(texts):
        return await ollama_embed(
            texts=texts,
            model="nomic-embed-text"  # Local embedding model
        )
    
    embedding_func = EmbeddingFunc(
        embedding_dim=768,  # Different dimension!
        max_token_size=2048,
        func=local_embedding_func
    )
    
    # Configure local LLM function
    def local_llm_func(prompt, **kwargs):
        return ollama_complete(
            prompt=prompt,
            model="llama3.1:8b",  # Local LLM model
            **kwargs
        )
    
    # Create target RAG instance
    target_rag = LightRAG(
        working_dir="./rag_storage_target",
        llm_model_func=local_llm_func,
        embedding_func=embedding_func,
        # Optimized for local performance
        chunk_token_size=512,
        entity_extract_max_gleaning=1,
        max_async=4,
        max_tokens_for_global_context=8192
    )
    
    return target_rag


async def import_knowledge_graph(package_path: str):
    """Import knowledge graph to Computer B"""
    print("=== COMPUTER B: Importing Knowledge Graph ===")
    
    # Setup target RAG
    target_rag = await setup_target_rag()
    
    # Create transfer instance
    transfer = KnowledgeGraphTransfer()
    
    # Configure import - optimized for local models and embedding regeneration
    import_config = TransferConfig(
        include_embeddings=False,     # Skip source embeddings (different model)
        include_llm_cache=True,       # Import LLM cache for efficiency
        include_documents=True,       # Import documents for re-processing
        verify_integrity=True,        # Verify transfer completed correctly
        regenerate_embeddings=True,   # Regenerate embeddings with local model
        batch_size=50                 # Smaller batches for local processing
    )
    
    # Execute import
    print("Importing knowledge graph...")
    success = await transfer.import_from_transfer(
        target_rag=target_rag,
        transfer_package_path=package_path,
        config=import_config
    )
    
    if success:
        print("Import completed successfully!")
        return target_rag
    else:
        print("Import failed!")
        return None


# ============================================================================
# Validation and Testing
# ============================================================================

async def validate_transfer(target_rag: LightRAG):
    """Validate the transferred knowledge graph"""
    print("=== Validating Transfer ===")
    
    from lightrag.transfer import TransferValidator
    
    validator = TransferValidator()
    
    # Configure validation
    validation_config = ValidationConfig(
        check_node_count=True,
        check_edge_count=True,
        check_embeddings=True,
        check_query_functionality=True,
        sample_queries=[
            "What are the main topics in this knowledge base?",
            "How are the key concepts related?",
            "Summarize the most important findings.",
        ],
        performance_threshold_seconds=30.0
    )
    
    # Note: We'd need source metadata for full validation
    # For this example, we'll do basic functionality testing
    print("Testing query functionality...")
    
    test_queries = [
        "What are the main entities in the knowledge graph?",
        "How are the concepts interconnected?",
        "What are the key relationships?"
    ]
    
    for i, query in enumerate(test_queries):
        try:
            print(f"\nTest Query {i+1}: {query}")
            
            # Test different modes
            for mode in ["naive", "local", "global"]:
                start_time = asyncio.get_event_loop().time()
                result = target_rag.query(query, param={"mode": mode})
                end_time = asyncio.get_event_loop().time()
                
                query_time = end_time - start_time
                print(f"  {mode.upper()} mode: {query_time:.2f}s")
                print(f"  Result: {result[:100]}...")
                
        except Exception as e:
            print(f"  Error in query {i+1}: {str(e)}")
    
    print("\nValidation completed!")


# ============================================================================
# Complete Transfer Workflow
# ============================================================================

async def complete_transfer_workflow():
    """Execute the complete transfer workflow"""
    print("Starting Complete Knowledge Graph Transfer Workflow")
    print("=" * 60)
    
    try:
        # Step 1: Export from Computer A
        package_path = await export_knowledge_graph()
        
        print(f"\nTransfer package ready: {package_path}")
        print("Now transfer this package to Computer B...")
        
        # Simulate transfer to Computer B
        print("\n" + "=" * 60)
        
        # Step 2: Import to Computer B
        target_rag = await import_knowledge_graph(package_path)
        
        if target_rag:
            # Step 3: Validate transfer
            await validate_transfer(target_rag)
            
            print("\n" + "=" * 60)
            print("Transfer workflow completed successfully!")
            print("\nNext steps:")
            print("1. Test query performance with your specific use cases")
            print("2. Adjust similarity thresholds if needed")
            print("3. Monitor embedding quality with real queries")
            print("4. Consider fine-tuning local models if performance is suboptimal")
            
        else:
            print("Transfer workflow failed during import phase")
            
    except Exception as e:
        print(f"Transfer workflow failed: {str(e)}")
        raise


# ============================================================================
# Utility Functions
# ============================================================================

def setup_environment():
    """Setup environment variables and directories"""
    # Create necessary directories
    os.makedirs("./rag_storage_source", exist_ok=True)
    os.makedirs("./rag_storage_target", exist_ok=True)
    os.makedirs("./documents/folder_X", exist_ok=True)
    os.makedirs("./additional_docs", exist_ok=True)

    # Set environment variables (customize as needed)
    os.environ.setdefault("OPENAI_API_KEY", "your-openai-api-key")

    # Optional: Setup custom PDF processor
    setup_custom_pdf_processor()

    print("Environment setup completed")


def setup_custom_pdf_processor():
    """Example of setting up a custom PDF processor"""

    # Example custom PDF processor
    class CustomPDFProcessor(BasePDFProcessor):
        """Custom PDF processor with enhanced text extraction"""

        def is_available(self) -> bool:
            try:
                import PyPDF2
                return True
            except ImportError:
                return False

        def extract_text(self, pdf_path) -> str:
            """Custom PDF text extraction with preprocessing"""
            import PyPDF2

            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text_content = []

                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    # Custom preprocessing
                    if page_text.strip():
                        # Remove excessive whitespace
                        cleaned_text = ' '.join(page_text.split())
                        text_content.append(cleaned_text)

                return "\n\n".join(text_content)

    # Uncomment to use custom processor
    # set_pdf_processor(CustomPDFProcessor())
    print("Custom PDF processor setup available (commented out)")


def cleanup_example():
    """Clean up example files"""
    import shutil
    
    cleanup_paths = [
        "./rag_storage_source",
        "./rag_storage_target", 
        "./transfer_package",
        "./transfer_package.tar.gz"
    ]
    
    for path in cleanup_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    
    print("Cleanup completed")


# ============================================================================
# Main Execution
# ============================================================================

async def main():
    """Main execution function"""
    print("LightRAG Knowledge Graph Transfer Example")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    try:
        # Run complete transfer workflow
        await complete_transfer_workflow()
        
    except KeyboardInterrupt:
        print("\nTransfer interrupted by user")
    except Exception as e:
        print(f"\nTransfer failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Optionally cleanup (comment out to keep files for inspection)
    # cleanup_example()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
