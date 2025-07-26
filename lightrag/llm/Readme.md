
# LLM Backends

LightRAG supports multiple LLM backends for both completions and embeddings:

## Available Backends

1. **OpenAI** (`llm/openai.py`):
   - Direct OpenAI API integration
   - Supports GPT models and OpenAI embeddings
   - See examples: [OpenAI Demo](../../examples/lightrag_openai_demo.py)

2. **Ollama** (`llm/ollama.py`):
   - Local model deployment
   - Supports various open-source models
   - See examples: [Ollama Demo](../../examples/lightrag_ollama_demo.py)

3. **Anthropic** (`llm/anthropic.py`):
   - Claude models integration
   - Includes Voyage AI embeddings support
   - See examples: [Anthropic Demo](../../examples/lightrag_anthropic_demo.py)

4. **HuggingFace** (`llm/hf.py`):
   - Local HuggingFace model deployment
   - Transformers library integration
   - See examples: [HuggingFace Demo](../../examples/unofficial-sample/lightrag_hf_demo.py)

5. **xAI Grok** (`llm/grok.py`):
   - Grok 3, Grok 4, and other xAI models
   - OpenAI-compatible API integration
   - See examples: [Grok Demo](../../examples/lightrag_grok_demo.py)

6. **Other Providers**:
   - Jina AI (`llm/jina.py`)
   - Zhipu AI (`llm/zhipu.py`)
   - LMDeploy (`llm/lmdeploy.py`)

## Usage

Each backend provides completion and embedding functions that can be used with LightRAG:

```python
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embedding

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=openai_embedding,
)
```

For detailed setup instructions and examples, see the individual demo files in the `examples/` directory.
