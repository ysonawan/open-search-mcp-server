# OpenSearch MCP Server - Code Search Agent

A comprehensive guide to setting up OpenSearch for code search, ingesting source code, and integrating it with an MCP (Model Context Protocol) Server for AI-powered code search capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installing OpenSearch](#installing-opensearch)
4. [Setting Up Indexes](#setting-up-indexes)
5. [Ingesting Code](#ingesting-code)
6. [Searching Code](#searching-code)
7. [OpenSearch Dashboard](#opensearch-dashboard)
8. [MCP Server Integration](#mcp-server-integration)
9. [Architecture](#architecture)

---

## Overview

This OpenSearch MCP Server enables AI agents to search through codebases using both semantic and keyword-based search. It provides:

- **Hybrid Search**: Combines semantic understanding (via embeddings) with keyword matching
- **Dual Indexes**: Text-only index for fast keyword search and vector-enabled index for semantic search
- **Repository Filtering**: Filter results by repository when needed
- **Code Metadata**: Retrieves file paths, languages, line numbers, and code snippets
- **Integration with AI**: Seamlessly integrates with Claude and other AI models via MCP protocol

---

## Prerequisites

Before you begin, ensure you have:

- **Docker**: For running OpenSearch
- **Docker Compose**: For orchestrating containers
- **Python 3.14+**: For running the ingestion scripts and MCP server
- **pip or uv**: Python package manager

---

## Installing OpenSearch

### Step 1: Create Environment File

Create a `.env` file in your project root with the OpenSearch initial admin password:

```bash
echo "OPENSEARCH_INITIAL_ADMIN_PASSWORD=<your-strong-password>" > .env
```

### Step 2: Start OpenSearch with Docker Compose

Download the docker image from the OpenSearch official website and use Docker Compose to start OpenSearch and OpenSearch Dashboards:

```bash
docker compose up -d
```

### Step 3: Verify OpenSearch is Running

Check that all containers are running:

```bash
docker ps
```

### Step 4: Health Check

Verify OpenSearch is responsive:

```bash
curl -k -u admin:<your-password> https://localhost:9200
```

Check cluster health:

```bash
curl -k -u admin:<your-password> https://localhost:9200/_cluster/health
```

Expected response:
```json
{
  "cluster_name": "opensearch-cluster",
  "status": "green",
  "timed_out": false,
  "number_of_nodes": 1,
  "number_of_data_nodes": 1,
  "active_primary_shards": 0,
  "active_shards": 0,
  "relocating_shards": 0,
  "initializing_shards": 0,
  "unassigned_shards": 0,
  "delayed_unassigned_shards": 0,
  "number_of_pending_tasks": 0,
  "number_of_in_flight_fetch": 0,
  "task_max_waiting_in_queue_ms": 0,
  "active_shards_percent_as_number": 100.0
}
```

---

## Setting Up Indexes

This guide uses two indexes for different use cases:

1. **code-search-text-only**: For keyword-based search
2. **code-search-with-vectors**: For hybrid search (keyword + semantic)

### Index Definition Files

#### code-search-text-only.json

```json
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "analysis": {
      "tokenizer": {
        "code_tokenizer": {
          "type": "pattern",
          "pattern": "[^a-zA-Z0-9_]"
        }
      },
      "filter": {
        "limit_token_length": {
          "type": "length",
          "max": 256
        }
      },
      "analyzer": {
        "code_analyzer": {
          "type": "custom",
          "tokenizer": "code_tokenizer",
          "filter": [
            "lowercase",
            "limit_token_length"
          ]
        }
      }
    }
  },
  "mappings": {
    "dynamic": "false",
    "properties": {
      "repo": {
        "type": "keyword"
      },
      "path": {
        "type": "keyword"
      },
      "language": {
        "type": "keyword"
      },
      "chunk_id": {
        "type": "integer"
      },
      "content": {
        "type": "text",
        "analyzer": "code_analyzer"
      }
    }
  }
}
```

#### code-search-with-vectors.json

```json
{
  "settings": {
    "index.knn": true,
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "analysis": {
      "tokenizer": {
        "code_tokenizer": {
          "type": "pattern",
          "pattern": "[^a-zA-Z0-9_]"
        }
      },
      "filter": {
        "limit_token_length": {
          "type": "length",
          "max": 256
        }
      },
      "analyzer": {
        "code_analyzer": {
          "type": "custom",
          "tokenizer": "code_tokenizer",
          "filter": [
            "lowercase",
            "limit_token_length"
          ]
        }
      }
    }
  },
  "mappings": {
    "dynamic": "false",
    "properties": {
      "repo": { "type": "keyword" },
      "path": { "type": "keyword" },
      "language": { "type": "keyword" },
      "chunk_id": { "type": "integer" },
      "start_line": { "type": "integer" },
      "end_line": { "type": "integer" },

      "content": {
        "type": "text",
        "analyzer": "code_analyzer"
      },

      "embedding": {
        "type": "knn_vector",
        "dimension": 384,
        "method": {
          "name": "hnsw",
          "engine": "nmslib",
          "space_type": "cosinesimil"
        }
      }
    }
  }
}
```

### Create Indexes

Create the text-only index:

```bash
curl -k -u admin:<your-password> \
  -X PUT "https://localhost:9200/code-search-text-only" \
  -H "Content-Type: application/json" \
  -d @code-search-text-only.json
```

Create the vector-enabled index:

```bash
curl -k -u admin:<your-password> \
  -X PUT https://localhost:9200/code-search-with-vectors \
  -H "Content-Type: application/json" \
  -d @code-search-with-vectors.json
```

### Verify Index Creation

Check the mappings for text-only index:

```bash
curl -k -u admin:<your-password> https://localhost:9200/code-search-text-only/_mapping?pretty
```

Check the mappings for vector-enabled index:

```bash
curl -k -u admin:<your-password> https://localhost:9200/code-search-with-vectors/_mapping?pretty
```

---

## Ingesting Code

### Python Dependencies

Install required dependencies:

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv pip install -r requirements.txt
```

### Ingestion Scripts

You'll need two ingestion scripts:

#### ingest-code-text-only.py

This script ingests code into the text-only index:

```python
import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers
from tree_sitter_languages import get_parser

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# OpenSearch configuration
# -----------------------------
opensearch_password = os.getenv("OPENSEARCH_INITIAL_ADMIN_PASSWORD")
if not opensearch_password:
    raise ValueError("OPENSEARCH_INITIAL_ADMIN_PASSWORD not found in .env file")

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", opensearch_password),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
)

INDEX_NAME = "code-search-text-only"

# -----------------------------
# Config
# -----------------------------
LANGUAGE_MAP = {
    ".java": "java",
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    '.php': 'php',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala',
}

SKIP_EXTENSIONS = (
    ".min.js", ".map", ".lock", ".zip", ".jar", ".class",
    ".png", ".jpg", ".jpeg", ".gif", ".pdf"
)

MAX_FILE_SIZE = 500_000        # 500 KB
LINE_CHUNK_SIZE = 200          # lines
LINE_CHUNK_OVERLAP = 40

# -----------------------------
# Helpers
# -----------------------------
def detect_language(path):
    for ext, lang in LANGUAGE_MAP.items():
        if path.endswith(ext):
            return lang
    return None

def should_skip(path):
    if path.lower().endswith(SKIP_EXTENSIONS):
        return True
    if os.path.getsize(path) > MAX_FILE_SIZE:
        return True
    return False

# -----------------------------
# Tree-sitter symbol extraction
# -----------------------------
SYMBOL_NODES = {
    "function_definition",
    "method_definition",
    "method_declaration",
    "function_declaration",
    "class_definition",
    "class_declaration"
}

def extract_symbols(code, language):
    parser = get_parser(language)
    tree = parser.parse(code.encode("utf8"))
    root = tree.root_node
    symbols = []

    def walk(node):
        if node.type in SYMBOL_NODES:
            start = node.start_byte
            end = node.end_byte
            symbols.append({
                "text": code[start:end],
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1
            })
        for child in node.children:
            walk(child)

    walk(root)
    return symbols

# -----------------------------
# Line fallback chunking
# -----------------------------
def line_chunks(code):
    lines = code.splitlines()
    chunks = []

    i = 0
    while i < len(lines):
        start = i
        end = min(i + LINE_CHUNK_SIZE, len(lines))
        chunk = "\n".join(lines[start:end])
        chunks.append({
            "text": chunk,
            "start_line": start + 1,
            "end_line": end
        })
        i += (LINE_CHUNK_SIZE - LINE_CHUNK_OVERLAP)

    return chunks

# -----------------------------
# Ingestion
# -----------------------------
def walk_repo(repo_path):
    repo_name = os.path.basename(os.path.abspath(repo_path))
    actions = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, repo_path)

            if should_skip(full_path):
                continue

            language = detect_language(full_path)
            if not language:
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
            except Exception:
                continue

            # 1️⃣ Symbol chunking
            chunks = extract_symbols(code, language)

            # 2️⃣ Fallback to line chunking
            if not chunks:
                chunks = line_chunks(code)

            for chunk_id, chunk in enumerate(chunks):
                actions.append({
                    "_index": INDEX_NAME,
                    "_source": {
                        "repo": repo_name,
                        "path": rel_path,
                        "language": language,
                        "chunk_id": chunk_id,
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "content": chunk["text"]
                    }
                })

            if len(actions) >= 500:
                helpers.bulk(client, actions)
                actions.clear()

    if actions:
        helpers.bulk(client, actions)

    print(f"Indexing complete for repo: {repo_name}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    walk_repo("<your repo path>")
```

#### ingest-code-with-vectors.py

This script ingests code with embeddings into the vector-enabled index:

```python
import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers
from tree_sitter_languages import get_parser
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# OpenSearch configuration
# -----------------------------
opensearch_password = os.getenv("OPENSEARCH_INITIAL_ADMIN_PASSWORD")
if not opensearch_password:
    raise ValueError("OPENSEARCH_INITIAL_ADMIN_PASSWORD not found in .env file")

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", opensearch_password),
    use_ssl=True,
    verify_certs=False,
    ssl_show_warn=False,
)

INDEX_NAME = "code-search-with-vectors"

# -----------------------------
# Embedding model (loaded once)
# -----------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# -----------------------------
# Config
# -----------------------------
LANGUAGE_MAP = {
    ".java": "java",
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    '.php': 'php',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala',
}

SKIP_EXTENSIONS = (
    ".min.js", ".map", ".lock", ".zip", ".jar", ".class",
    ".png", ".jpg", ".jpeg", ".gif", ".pdf"
)

MAX_FILE_SIZE = 500_000        # 500 KB
LINE_CHUNK_SIZE = 200          # lines
LINE_CHUNK_OVERLAP = 40

# -----------------------------
# Helpers
# -----------------------------
def detect_language(path):
    for ext, lang in LANGUAGE_MAP.items():
        if path.endswith(ext):
            return lang
    return None

def should_skip(path):
    if path.lower().endswith(SKIP_EXTENSIONS):
        return True
    if os.path.getsize(path) > MAX_FILE_SIZE:
        return True
    return False

# -----------------------------
# Tree-sitter symbol extraction
# -----------------------------
SYMBOL_NODES = {
    "function_definition",
    "method_definition",
    "method_declaration",
    "function_declaration",
    "class_definition",
    "class_declaration"
}

def extract_symbols(code, language):
    parser = get_parser(language)
    tree = parser.parse(code.encode("utf8"))
    root = tree.root_node
    symbols = []

    def walk(node):
        if node.type in SYMBOL_NODES:
            start = node.start_byte
            end = node.end_byte
            symbols.append({
                "text": code[start:end],
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1
            })
        for child in node.children:
            walk(child)

    walk(root)
    return symbols

# -----------------------------
# Line fallback chunking
# -----------------------------
def line_chunks(code):
    lines = code.splitlines()
    chunks = []

    i = 0
    while i < len(lines):
        start = i
        end = min(i + LINE_CHUNK_SIZE, len(lines))
        chunk = "\n".join(lines[start:end])
        chunks.append({
            "text": chunk,
            "start_line": start + 1,
            "end_line": end
        })
        i += (LINE_CHUNK_SIZE - LINE_CHUNK_OVERLAP)

    return chunks

# -----------------------------
# Ingestion
# -----------------------------
def walk_repo(repo_path):
    repo_name = os.path.basename(os.path.abspath(repo_path))
    actions = []

    for root, _, files in os.walk(repo_path):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, repo_path)

            if should_skip(full_path):
                continue

            language = detect_language(full_path)
            if not language:
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    code = f.read()
            except Exception:
                continue

            # 1️⃣ Symbol chunking
            chunks = extract_symbols(code, language)

            # 2️⃣ Fallback to line chunking
            if not chunks:
                chunks = line_chunks(code)

            for chunk_id, chunk in enumerate(chunks):
                # ---- Embedding generation ----
                embedding = embedding_model.encode(
                    chunk["text"],
                    normalize_embeddings=True
                ).tolist()

                actions.append({
                    "_index": INDEX_NAME,
                    "_source": {
                        "repo": repo_name,
                        "path": rel_path,
                        "language": language,
                        "chunk_id": chunk_id,
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "content": chunk["text"],
                        "embedding": embedding
                    }
                })

            if len(actions) >= 500:
                helpers.bulk(client, actions)
                actions.clear()

    if actions:
        helpers.bulk(client, actions)

    print(f"Indexing complete for repo: {repo_name}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    walk_repo("<your repo path>")
```

### Run Ingestion

Ingest code into text-only index:

```bash
python ingest-code-text-only.py
```

Ingest code with embeddings into vector index:

```bash
python ingest-code-with-vectors.py
```

### Verify Ingestion

Count documents in text-only index:

```bash
curl -k -u admin:<your-password> https://localhost:9200/code-search-text-only/_count
```

Count documents in vector-enabled index:

```bash
curl -k -u admin:<your-password> https://localhost:9200/code-search-with-vectors/_count
```

Expected response:
```json
{
  "count": 1234,
  "_shards": {
    "total": 2,
    "successful": 2,
    "skipped": 0,
    "failed": 0
  }
}
```

---

## Searching Code

### CLI Search on Text-Only Index

Search for specific function names:

```bash
curl -k -u admin:<your-password> \
  -X GET https://localhost:9200/code-search-text-only/_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "match": {
        "content": "getGTTs"
      }
    }
  }'
```

Search for API endpoints:

```bash
curl -k -u admin:<your-password> \
  -X GET https://localhost:9200/code-search-text-only/_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "match": {
        "content": "/shops/{shopId}"
      }
    }
  }'
```

### CLI Search on Vector Index

Search for semantic matches:

```bash
curl -k -u admin:<your-password> \
  -X GET https://localhost:9200/code-search-with-vectors/_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "match": {
        "content": "retry mechanism"
      }
    }
  }'
```

### Advanced Query: Hybrid Search

Combine keyword and semantic search (example for `code-search-with-vectors`):

```bash
curl -k -u admin:<your-password> \
  -X GET https://localhost:9200/code-search-with-vectors/_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "bool": {
        "should": [
          {
            "match": {
              "content": "authentication handler"
            }
          }
        ]
      }
    }
  }'
```

---

## OpenSearch Dashboard

### Access Dashboard

1. Open your browser and navigate to: `https://localhost:5601`
2. Log in with:
   - Username: `admin`
   - Password: `<your-password>`

### Dashboard Features

#### 1. Index Management

- Navigate to **Stack Management** → **Index Management**
- View all indexes and their health status
- Monitor shard allocation and document count

#### 2. Dev Tools

- Go to **Management** → **Dev Tools**
- Use the Console to run Elasticsearch/OpenSearch queries
- Test your search queries interactively

#### 3. Search Queries in Dashboard

Example search in the Console:

```json
GET /code-search-text-only/_search
{
  "query": {
    "match": {
      "content": "database"
    }
  },
  "size": 20
}
```

#### 4. Visualization

- Create visualizations from search results
- Build dashboards to monitor code search metrics
- Track ingestion progress and document counts

---

## MCP Server Integration

### What is the MCP Server?

The **Model Context Protocol (MCP) Server** is an integration layer that connects your OpenSearch instance with AI models like Claude. It enables:

- Natural language code search queries
- Automatic query translation to OpenSearch queries
- Result formatting and ranking
- Integration with AI-powered code analysis tools

### Architecture

```
┌─────────────┐
│   Claude/   │
│     AI      │
└──────┬──────┘
       │ (MCP Protocol)
       ▼
┌──────────────────────┐
│  MCP Server (main.py)│
│ - search_code()      │
│ - code_search_agent()│
└──────┬───────────────┘
       │ (HTTP/REST)
       ▼
┌──────────────────────┐
│   OpenSearch         │
│  - Vector Search     │
│  - Keyword Search    │
└──────────────────────┘
```

### Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv sync
```

### Configuration

The MCP server connects to OpenSearch using:

- **Host**: `localhost`
- **Port**: `9200`
- **Auth**: Admin credentials (configured in main.py)
- **Index**: `code-search-with-vectors` (hybrid search index)

### Running the MCP Server

Start the MCP server:

```bash
python main.py
```

Expected output:
```
Starting MCP Server on http://localhost:8003
```

### Server Capabilities

The MCP server provides:

#### 1. search_code Tool

**Parameters:**
- `query` (string, required): Natural language search query
- `repo` (string, optional): Filter results by repository

**Returns:**
- List of code snippets with metadata:
  - `score`: Relevance score (0-1)
  - `repo`: Repository name
  - `path`: File path
  - `language`: Programming language
  - `start_line`: Starting line number
  - `end_line`: Ending line number
  - `content`: Code snippet

**Example Query:**
```
Find all functions that handle authentication
```

**Expected Response:**
```json
[
  {
    "score": 0.95,
    "repo": "my-repo",
    "path": "src/auth/handler.py",
    "language": "python",
    "start_line": 42,
    "end_line": 68,
    "content": "def handle_authentication(...)"
  }
]
```

#### 2. code_search_agent Prompt

Provides system instructions for the AI agent on how to:
- Use the search_code tool effectively
- Format responses with proper context
- Never fabricate code snippets
- Handle edge cases (no results found)

### Integration with Claude

Once the MCP server is running, configure your MCP client (e.g., Claude desktop app) to connect:

1. Add server configuration pointing to `http://localhost:8003`
2. Authenticate with appropriate credentials
3. Use natural language to search code

Example conversation:
```
User: "Find functions that implement retry logic"
Claude: [Uses search_code tool] "I found 3 code snippets that implement retry mechanisms..."
```

### Search Strategy

The MCP server uses **hybrid search**:

1. **Semantic Search**: Uses embeddings to understand query intent
2. **Keyword Search**: Performs exact keyword matching with 2x boost
3. **Ranking**: Combines scores from both methods
4. **Filtering**: Optionally filters by repository

---

## Maintenance

### Delete Indexes

To completely reset and delete indexes:

```bash
curl -k -u admin:<your-password> -X DELETE https://localhost:9200/code-search-text-only
curl -k -u admin:<your-password> -X DELETE https://localhost:9200/code-search-with-vectors
```

### Stop OpenSearch

Stop and remove containers:

```bash
docker compose down
```

Remove volumes (clears all data):

```bash
docker compose down -v
```

### Backup Data

Create a backup of your OpenSearch data:

```bash
docker compose exec opensearch curl -k -u admin:<your-password> \
  -X POST "https://localhost:9200/_snapshot/backup" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "fs",
    "settings": {
      "location": "/backups"
    }
  }'
```

---

## Troubleshooting

### Issue: Connection Refused

**Solution**: Ensure OpenSearch is running and healthy:
```bash
docker ps
curl -k -u admin:<your-password> https://localhost:9200
```

### Issue: Authentication Failed

**Solution**: Verify credentials match those in `.env` file and check logs:
```bash
docker compose logs opensearch
```

### Issue: No Results in Search

**Solution**: Verify data was ingested:
```bash
curl -k -u admin:<your-password> https://localhost:9200/code-search-text-only/_count
```

### Issue: MCP Server Connection Error

**Solution**: Check if server is running on port 8003:
```bash
curl http://localhost:8003
```

---

## Next Steps

1. **Customize Ingestion**: Modify ingest scripts to match your codebase structure
2. **Tune Search Parameters**: Adjust chunk sizes and boost factors for better results
3. **Monitor Performance**: Use OpenSearch Dashboard to track query performance
4. **Scale Up**: Configure multiple shards and replicas for larger codebases
5. **Advanced Analytics**: Build custom dashboards to analyze code metrics

---

## References

- [OpenSearch Documentation](https://opensearch.org/docs/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastMCP Documentation](https://fastmcp.dev/)
- [OpenSearch Python Client](https://opensearch-project.github.io/opensearch-py/)

---

## License

This project is provided as-is for code search and AI integration purposes.
