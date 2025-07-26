import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.grok import grok_4_complete, grok_3_complete, grok_3_mini_complete
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# Set up environment variables
# You need to set XAI_API_KEY for Grok models
# You need to set OPENAI_API_KEY for embeddings (since xAI doesn't provide embeddings)
if not os.environ.get("XAI_API_KEY"):
    raise ValueError("Please set XAI_API_KEY environment variable")

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY environment variable for embeddings")

WORKING_DIR = "./grok_rag_storage"

async def main():
    # Create working directory
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    # Initialize pipeline status
    initialize_pipeline_status(WORKING_DIR)

    # Initialize LightRAG with Grok 4 and OpenAI embeddings
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=grok_4_complete,  # Use Grok 4 for completions
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8191,
            func=openai_embed,  # Use OpenAI embeddings as fallback
        ),
    )

    # Example text to insert
    sample_text = """
    xAI is an AI company founded by Elon Musk in 2023. The company's flagship product is Grok, 
    a large language model designed to be truthful and helpful. Grok is trained on a diverse 
    dataset and is designed to answer questions with wit and humor, while avoiding harmful or 
    biased responses.
    
    Grok 4 is the latest version of the model, featuring advanced reasoning capabilities, 
    vision support, function calling, and structured outputs. It has a context window of 
    256,000 tokens and supports multiple modalities.
    
    The xAI API is compatible with OpenAI's API format, making it easy for developers to 
    integrate Grok into their existing applications. The API supports various models including 
    Grok 3, Grok 3 Mini, and Grok 4, each optimized for different use cases and performance 
    requirements.
    """

    print("Inserting sample text...")
    await rag.ainsert(sample_text)

    # Example queries
    queries = [
        "What is xAI and who founded it?",
        "Tell me about Grok 4's capabilities",
        "How is the xAI API compatible with existing tools?",
    ]

    print("\n" + "="*50)
    print("QUERY RESULTS")
    print("="*50)

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        # Naive search
        print("Naive search:")
        naive_result = await rag.aquery(query, param=QueryParam(mode="naive"))
        print(naive_result)
        
        print("\nLocal search:")
        local_result = await rag.aquery(query, param=QueryParam(mode="local"))
        print(local_result)
        
        print("\nGlobal search:")
        global_result = await rag.aquery(query, param=QueryParam(mode="global"))
        print(global_result)
        
        print("\nHybrid search:")
        hybrid_result = await rag.aquery(query, param=QueryParam(mode="hybrid"))
        print(hybrid_result)
        
        print("\n" + "="*50)


async def demo_different_models():
    """Demo showing how to use different Grok models."""
    print("\n" + "="*50)
    print("DIFFERENT GROK MODELS DEMO")
    print("="*50)
    
    # Test prompt
    prompt = "Explain quantum computing in simple terms."
    
    # Test Grok 4
    print("\nGrok 4 Response:")
    print("-" * 20)
    grok4_response = await grok_4_complete(prompt)
    print(grok4_response)
    
    # Test Grok 3
    print("\nGrok 3 Response:")
    print("-" * 20)
    grok3_response = await grok_3_complete(prompt)
    print(grok3_response)
    
    # Test Grok 3 Mini
    print("\nGrok 3 Mini Response:")
    print("-" * 20)
    grok3_mini_response = await grok_3_mini_complete(prompt)
    print(grok3_mini_response)


if __name__ == "__main__":
    print("Starting LightRAG with xAI Grok Demo...")
    print("Make sure you have set XAI_API_KEY and OPENAI_API_KEY environment variables")
    
    # Run the main demo
    asyncio.run(main())
    
    # Run the different models demo
    asyncio.run(demo_different_models())
    
    print("\nDemo completed!")
