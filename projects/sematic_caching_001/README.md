# Semantic Caching with Deep Research Agent

A comprehensive semantic caching system powered by Redis vector database and LangGraph, featuring an intelligent research agent that progressively builds knowledge through cached queries.

## 🌟 Features

- **Semantic Caching**: Intelligent caching using vector embeddings to match semantically similar queries
- **Deep Research Agent**: LangGraph-powered agentic workflow with query decomposition and synthesis
- **Knowledge Base Management**: Dynamic knowledge base creation from web content using Tavily
- **Multiple Reranking Strategies**: Cross-encoder and LLM-based reranking for improved cache hit accuracy
- **Interactive Demo**: Gradio-based web interface for real-time demonstration
- **Performance Analytics**: Built-in evaluation and visualization tools

## 📋 Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- OpenAI API key (for OpenAI models)
- Google API key (for Gemini models)
- Tavily API key (for web content extraction)

## 🚀 Getting Started

### 2. Set Up Environment Variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit the `.env` file and add your API keys:

```env
# LLM Provider Configuration
# Choose your LLM provider: 'openai' or 'gemini' (default: openai)
LLM_PROVIDER=openai

# API Keys
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

> **Get your API keys from:**
>
> - OpenAI: https://platform.openai.com/account/api-keys
> - Google Gemini: https://aistudio.google.com/apikey
> - Tavily: https://www.tavily.com/developers

> **Note**: You only need the API key for the provider you plan to use. Set `LLM_PROVIDER=openai` to use OpenAI models or `LLM_PROVIDER=gemini` to use Google Gemini models.

### 3. Initialize Docker (Redis)

Start the Redis vector database using Docker Compose:

```bash
docker-compose up -d
```

This will:

- Start Redis 8.0.3 on port 6379
- Create a persistent volume for data storage
- Configure automatic restarts

To verify Redis is running:

```bash
docker ps
```

To stop Redis:

```bash
docker-compose down
```

### 4. Install Dependencies

Create a virtual environment and install required packages:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Start the Application

Launch the Gradio demo interface:

```bash
python -c "from agent import launch_demo, initialize_agent; from cache.wrapper import SemanticCacheWrapper; from cache.config import config; cache = SemanticCacheWrapper.from_config(config); from langgraph.graph import StateGraph; from agent import WorkflowState; workflow = StateGraph(WorkflowState).compile(); launch_demo(workflow, cache)"
```

Or use the Jupyter notebook for interactive exploration:

```bash
jupyter notebook semantic_caching.ipynb
```

The Gradio interface will open in your browser, typically at `http://localhost:7860`.

## 📁 Project Structure

```
sematic_caching_project/
├── agent/                      # Deep research agent implementation
│   ├── __init__.py            # Agent initialization and workflow
│   ├── demo.py                # Gradio demo interface
│   ├── nodes.py               # LangGraph node functions
│   ├── edges.py               # Workflow routing logic
│   ├── tools.py               # Agent tools (knowledge base search)
│   └── knowledge_base_utils.py # Knowledge base management
├── cache/                      # Semantic caching system
│   ├── wrapper.py             # Main cache wrapper with reranking
│   ├── config.py              # Configuration management
│   ├── cross_encoder.py       # Cross-encoder reranking
│   ├── llm_evaluator.py       # LLM-based reranking
│   ├── fuzzy_cache.py         # Fuzzy string matching cache
│   ├── evals.py               # Evaluation utilities
│   └── vis.py                 # Visualization tools
├── data/                       # Sample datasets
│   ├── faq_seed.csv           # FAQ seed data
│   └── test_dataset.csv       # Test queries
├── docker-compose.yml          # Redis container configuration
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project metadata
├── .env.example               # Environment variables template
└── semantic_caching.ipynb      # Interactive Jupyter notebook
```

## 🔧 Configuration

The semantic cache can be configured via environment variables or the `cache/config.py` file:

| Variable                   | Default                  | Description                                |
| -------------------------- | ------------------------ | ------------------------------------------ |
| `LLM_PROVIDER`             | `openai`                 | LLM provider to use (`openai` or `gemini`) |
| `REDIS_URL`                | `redis://localhost:6379` | Redis connection URL                       |
| `CACHE_NAME`               | `semantic-cache`         | Cache index name                           |
| `CACHE_DISTANCE_THRESHOLD` | `0.3`                    | Maximum semantic distance for cache hits   |
| `CACHE_TTL_SECONDS`        | `3600`                   | Time-to-live for cache entries (1 hour)    |
| `OPENAI_API_KEY`           | -                        | OpenAI API key (required if using OpenAI)  |
| `GOOGLE_API_KEY`           | -                        | Google API key (required if using Gemini)  |
| `TAVILY_API_KEY`           | -                        | Tavily API key (required)                  |

## 💡 Usage Examples

### Using the Gradio Demo

1. **Load Web Content**: Enter a URL to extract and index content
2. **Ask Questions**: Query the content using natural language
3. **View Performance**: Monitor cache hits and LLM call reduction

### Programmatic Usage

```python
from cache.wrapper import SemanticCacheWrapper
from cache.config import config

# Initialize semantic cache
cache = SemanticCacheWrapper.from_config(config)

# Store question-answer pairs
cache.hydrate_from_pairs([
    ("What is semantic caching?", "Semantic caching uses vector embeddings..."),
    ("How does it work?", "It compares query embeddings to find similar cached responses...")
])

# Check cache for similar queries
results = cache.check("Explain semantic caching", num_results=3)
if results.matches:
    print(f"Cache hit! Answer: {results.matches[0].response}")
```

### Using the Research Agent

```python
from agent import initialize_agent, run_agent
from cache.wrapper import SemanticCacheWrapper
from cache.config import config

# Initialize components
cache = SemanticCacheWrapper.from_config(config)
# ... initialize knowledge base and embeddings ...

# Run agent with caching
result = run_agent(workflow_app, "What are the main findings?", enable_caching=True)
print(result['final_response'])
```

### Model Selection

Switch between OpenAI and Gemini models using environment variables:

```python
import os

# Use OpenAI models (default)
os.environ["LLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "your_openai_key"

# Or use Gemini models
os.environ["LLM_PROVIDER"] = "gemini"
os.environ["GOOGLE_API_KEY"] = "your_google_key"

# The model factory will automatically use the selected provider
from cache.model_factory import get_llm
llm = get_llm("analysis")  # Returns OpenAI or Gemini based on LLM_PROVIDER
```

You can also set the provider in your `.env` file:

```bash
# .env file
LLM_PROVIDER=gemini  # or 'openai'
GOOGLE_API_KEY=your_google_api_key_here
```

## 🧪 Advanced Features

### Reranking Strategies

The cache supports multiple reranking strategies to improve accuracy:

1. **Cross-Encoder Reranking**: Uses a cross-encoder model for precise relevance scoring
2. **LLM-Based Reranking**: Leverages LLMs for context-aware filtering

```python
from cache.cross_encoder import create_cross_encoder_reranker

# Register a reranker
reranker = create_cross_encoder_reranker(threshold=0.5)
cache.register_reranker(reranker)
```

### Knowledge Base Management

```python
from agent.knowledge_base_utils import KnowledgeBaseManager
import redis

redis_client = redis.Redis.from_url("redis://localhost:6379")
kb_manager = KnowledgeBaseManager(redis_client)

# Create knowledge base from text
success, message, index = kb_manager.create_knowledge_base(
    source_id="my-document",
    content="Your text content here...",
    chunk_size=4500,
    chunk_overlap=250
)
```

## 📊 Performance Monitoring

The system tracks:

- Cache hit rate
- LLM call reduction
- Token savings
- Response time improvements

View metrics in the Gradio demo's Performance Log section.

## 🛠️ Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests (if available)
pytest
```

### Code Formatting

```bash
black .
```

## 🐛 Troubleshooting

### Redis Connection Issues

```bash
# Check if Redis is running
docker ps

# View Redis logs
docker-compose logs redis

# Restart Redis
docker-compose restart redis
```

### API Key Errors

Ensure your `.env` file contains valid API keys and is in the project root directory.

### Import Errors

Make sure you've activated the virtual environment and installed all dependencies:

```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 📚 Learn More

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Redis Vector Similarity](https://redis.io/docs/stack/search/reference/vectors/)
- [Sentence Transformers](https://www.sbert.net/)
- [Tavily API](https://docs.tavily.com/)

## 📄 License

This project is part of a semantic caching course demonstrating advanced caching techniques with vector embeddings.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**Built with**: Python, Redis, LangChain, LangGraph, Gradio, Sentence Transformers
