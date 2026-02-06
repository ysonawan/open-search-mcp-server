# OpenSearch MCP Server - Code Search Agent

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-2.14.2-green.svg)](https://fastmcp.dev/)
[![OpenSearch](https://img.shields.io/badge/OpenSearch-2.x-orange.svg)](https://opensearch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A powerful **Model Context Protocol (MCP) server** that enables AI agents like Claude to search through codebases using hybrid search—combining semantic understanding with keyword matching. This server bridges OpenSearch and AI models, providing intelligent code search capabilities through natural language queries.

## Features

- 🔍 **Hybrid Search**: Semantic (embedding-based) + keyword search for best results
- 🚀 **Fast & Scalable**: Built on OpenSearch with efficient indexing
- 🤖 **AI-Native**: Direct integration with Claude and other AI models via MCP
- 📦 **Easy Setup**: Docker-based OpenSearch deployment
- 🎯 **Smart Chunking**: Tree-sitter based symbol extraction with line-based fallback
- 🔧 **Flexible**: Support for 15+ programming languages
- 📊 **Observable**: Built-in OpenSearch Dashboard for monitoring

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Prerequisites](#prerequisites)
4. [Installing OpenSearch](#installing-opensearch)
5. [Setting Up Indexes](#setting-up-indexes)
6. [Ingesting Code](#ingesting-code)
7. [MCP Server Setup](#mcp-server-setup)
8. [Claude Desktop Integration](#claude-desktop-integration)
9. [Searching Code](#searching-code)
10. [OpenSearch Dashboard](#opensearch-dashboard)
11. [Architecture](#architecture)
12. [Development](#development)
13. [Troubleshooting](#troubleshooting)
14. [Contributing](#contributing)

---

## Overview

This OpenSearch MCP Server enables AI agents to search through codebases using both semantic and keyword-based search. It provides:

- **Hybrid Search**: Combines semantic understanding (via embeddings) with keyword matching
- **Dual Indexes**: Text-only index for fast keyword search and vector-enabled index for semantic search
- **Repository Filtering**: Filter results by repository when needed
- **Code Metadata**: Retrieves file paths, languages, line numbers, and code snippets
- **Integration with AI**: Seamlessly integrates with Claude and other AI models via MCP protocol

---

## Quick Start

Get up and running in 5 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/ysonawan/open-search-mcp-server.git
cd open-search-mcp-server

# 2. Set up environment
echo "OPENSEARCH_INITIAL_ADMIN_PASSWORD=YourStrongPassword123!" > .env

# 3. Start OpenSearch
cd open-search
docker compose up -d

# 4. Create indexes
curl -k -u admin:YourStrongPassword123! -X PUT \
  https://localhost:9200/code-search-with-vectors \
  -H "Content-Type: application/json" \
  -d @code-search-with-vectors.json

# 5. Install Python dependencies
cd ..
pip install -r requirements.txt
# or use uv: uv sync

# 6. Ingest your code (edit path in script first)
python ingest-code/ingest-code-with-vectors.py

# 7. Start MCP server
python main.py
```

Now configure Claude Desktop (see [Claude Desktop Integration](#claude-desktop-integration)) to start using natural language code search!

---

## Prerequisites

Before you begin, ensure you have:

- **Docker**: For running OpenSearch ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: For orchestrating containers
- **Python 3.12+**: For running the ingestion scripts and MCP server
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

## MCP Server Setup

### What is the MCP Server?

The **Model Context Protocol (MCP) Server** is an integration layer that connects your OpenSearch instance with AI models like Claude. It enables:

- Natural language code search queries
- Automatic query translation to OpenSearch queries
- Result formatting and ranking
- Integration with AI-powered code analysis tools

### Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv sync
```

**Note**: On first run, the sentence-transformers library will download the `all-MiniLM-L6-v2` model (~90MB). This is a one-time download.

### Configuration

The MCP server reads configuration from a `.env` file in the project root:

```bash
OPENSEARCH_INITIAL_ADMIN_PASSWORD=YourStrongPassword123!
```

The server connects to OpenSearch using:

- **Host**: `localhost`
- **Port**: `9200`
- **Auth**: Admin credentials from `.env`
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

The server will be ready to accept connections from MCP clients.

### Server Capabilities

The MCP server provides two main capabilities:

#### 1. `search_code` Tool

**Parameters:**
- `query` (string, required): Natural language search query
- `repo` (string, optional): Filter results by repository name

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

**Example Response:**
```json
[
  {
    "score": 0.95,
    "repo": "my-app",
    "path": "src/auth/handler.py",
    "language": "python",
    "start_line": 42,
    "end_line": 68,
    "content": "def handle_authentication(request):\n    ..."
  }
]
```

#### 2. `code_search_agent` Prompt

Provides system instructions for the AI agent on how to:
- Use the search_code tool effectively
- Format responses with proper context
- Never fabricate code snippets
- Handle edge cases (no results found)

### Search Strategy

The MCP server uses **hybrid search** combining:

1. **Semantic Search**: Uses embeddings to understand query intent
2. **Keyword Search**: Performs exact keyword matching with 2x boost
3. **Ranking**: Combines scores from both methods
4. **Filtering**: Optionally filters by repository

---

## Claude Desktop Integration

To use this MCP server with Claude Desktop, you need to configure Claude to connect to your local MCP server.

### Step 1: Locate Claude Desktop Config

The configuration file location depends on your operating system:

- **MacOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

### Step 2: Configure MCP Server

Add the following configuration to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "code-search": {
      "command": "python",
      "args": ["/absolute/path/to/open-search-mcp-server/main.py"],
      "env": {
        "OPENSEARCH_INITIAL_ADMIN_PASSWORD": "YourStrongPassword123!"
      }
    }
  }
}
```

**Important**: Replace `/absolute/path/to/open-search-mcp-server/` with the actual absolute path to your project directory.

### Step 3: Alternative Configuration with UV

If you're using `uv`, you can configure it like this:

```json
{
  "mcpServers": {
    "code-search": {
      "command": "uv",
      "args": ["run", "main.py"],
      "cwd": "/absolute/path/to/open-search-mcp-server",
      "env": {
        "OPENSEARCH_INITIAL_ADMIN_PASSWORD": "YourStrongPassword123!"
      }
    }
  }
}
```

### Step 4: Restart Claude Desktop

After updating the configuration:

1. Quit Claude Desktop completely
2. Restart the application
3. The MCP server will automatically start when Claude Desktop launches

### Step 5: Verify Connection

To verify the connection is working:

1. Open Claude Desktop
2. Start a new conversation
3. Try a query like: "Search for authentication functions in my codebase"
4. Claude should use the `search_code` tool to query your indexed code

### Usage Examples

Once configured, you can use natural language to search your code:

**Example 1: Find specific functionality**
```
User: "Find all database connection handling code"
Claude: [Uses search_code tool and returns relevant code snippets]
```

**Example 2: Search in specific repository**
```
User: "Search for error handling in the backend repository"
Claude: [Uses search_code with repo filter and returns results]
```

**Example 3: Understand code patterns**
```
User: "Show me how we implement caching"
Claude: [Searches and explains the caching implementation based on actual code]
```

### Troubleshooting MCP Connection

If Claude can't connect to the MCP server:

1. **Check OpenSearch is running**:
   ```bash
   curl -k -u admin:YourStrongPassword123! https://localhost:9200/_cluster/health
   ```

2. **Verify Python/UV is in PATH**:
   ```bash
   which python
   # or
   which uv
   ```

3. **Test MCP server manually**:
   ```bash
   python main.py
   # Should show: Starting MCP Server on http://localhost:8003
   ```

4. **Check Claude Desktop logs**:
   - MacOS: `~/Library/Logs/Claude/`
   - Windows: `%APPDATA%\Claude\logs\`
   - Linux: `~/.config/Claude/logs/`

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Claude Desktop                          │
│                   (AI Assistant Interface)                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ MCP Protocol (stdio/http)
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   MCP Server (main.py)                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tools:                                              │   │
│  │  • search_code(query, repo)                         │   │
│  │                                                       │   │
│  │  Prompts:                                            │   │
│  │  • code_search_agent()                              │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Components:                                         │   │
│  │  • SentenceTransformer (all-MiniLM-L6-v2)          │   │
│  │  • OpenSearch Python Client                         │   │
│  └─────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTPS/REST API
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      OpenSearch Cluster                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Index: code-search-with-vectors                    │   │
│  │  • Hybrid Search (KNN + BM25)                       │   │
│  │  • Vector Embeddings (384 dimensions)               │   │
│  │  • Custom Code Analyzer                             │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Index: code-search-text-only                       │   │
│  │  • Keyword Search (BM25)                            │   │
│  │  • Fast token-based matching                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
                            │ Bulk API
                            │
┌───────────────────────────┴─────────────────────────────────┐
│                   Ingestion Scripts                          │
│  • ingest-code-with-vectors.py                              │
│  • ingest-code-text-only.py                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • Tree-sitter parsing                               │   │
│  │  • Symbol extraction (functions, classes)            │   │
│  │  • Line-based chunking fallback                      │   │
│  │  • Embedding generation                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

#### 1. Ingestion Pipeline

```
Local Codebase
    │
    ├─→ Walk repository files
    │
    ├─→ Filter by extension & size
    │
    ├─→ Parse with tree-sitter
    │       │
    │       ├─→ Extract symbols (functions, classes)
    │       └─→ Fallback to line-based chunks
    │
    ├─→ Generate embeddings (SentenceTransformer)
    │
    └─→ Bulk index to OpenSearch
            │
            ├─→ code-search-text-only (keyword only)
            └─→ code-search-with-vectors (hybrid)
```

#### 2. Search Pipeline

```
User Query (Natural Language)
    │
    ├─→ Claude Desktop
    │
    ├─→ MCP Server (search_code tool)
    │       │
    │       ├─→ Generate query embedding
    │       │
    │       └─→ Build hybrid query:
    │           • KNN search on embeddings (k=50)
    │           • BM25 match on content (boost=2x)
    │           • Optional repo filter
    │
    ├─→ OpenSearch executes query
    │
    ├─→ Return top 10 results with scores
    │
    └─→ Format and display in Claude
```

### Component Details

#### MCP Server (`main.py`)
- **Framework**: FastMCP
- **Port**: 8003 (HTTP transport)
- **Model**: all-MiniLM-L6-v2 (384-dim embeddings)
- **Index**: code-search-with-vectors

#### OpenSearch
- **Version**: 2.x
- **Deployment**: Docker Compose (2 nodes)
- **Dashboard**: Port 5601
- **API**: Port 9200

#### Ingestion
- **Languages**: Java, Python, JavaScript, TypeScript, Go, Ruby, Rust, C, C++, C#, PHP, Swift, Kotlin, Scala
- **Chunking**: Symbol-based (preferred) or line-based (200 lines, 40 overlap)
- **Max File Size**: 500KB
- **Batch Size**: 500 documents

---

## Development

### Setting Up Development Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ysonawan/open-search-mcp-server.git
   cd open-search-mcp-server
   ```

2. **Install dependencies**:
   ```bash
   # Using pip
   pip install -r requirements.txt

   # Or using uv (recommended)
   uv sync
   ```

3. **Set up pre-commit hooks** (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Project Structure

```
open-search-mcp-server/
├── main.py                          # MCP server implementation
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project metadata
├── .env-example                     # Environment template
├── ingest-code/
│   ├── ingest-code-text-only.py    # Text-only ingestion
│   └── ingest-code-with-vectors.py # Vector ingestion
└── open-search/
    ├── docker-compose.yml           # OpenSearch setup
    ├── code-search-text-only.json   # Text index schema
    └── code-search-with-vectors.json # Vector index schema
```

### Customizing Ingestion

#### Adding New Language Support

Edit the `LANGUAGE_MAP` in ingestion scripts:

```python
LANGUAGE_MAP = {
    ".java": "java",
    ".py": "python",
    # Add your language
    ".elm": "elm",
    ".ex": "elixir",
}
```

#### Adjusting Chunk Sizes

Modify chunking parameters:

```python
LINE_CHUNK_SIZE = 200      # Number of lines per chunk
LINE_CHUNK_OVERLAP = 40    # Overlap between chunks
MAX_FILE_SIZE = 500_000    # Maximum file size (bytes)
```

#### Customizing Search Behavior

Edit search parameters in `main.py`:

```python
body = {
    "size": 10,  # Number of results (increase for more)
    "query": {
        "bool": {
            "should": [
                {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": 50  # Candidate pool size
                        }
                    }
                },
                {
                    "match": {
                        "content": {
                            "query": query,
                            "boost": 2  # Keyword boost factor
                        }
                    }
                }
            ]
        }
    }
}
```

### Testing

#### Test MCP Server Locally

```bash
# Start server
python main.py

# In another terminal, test with curl
curl -X POST http://localhost:8003 \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "search_code",
    "arguments": {
      "query": "authentication"
    }
  }'
```

#### Test OpenSearch Directly

```bash
# Health check
curl -k -u admin:YourPassword https://localhost:9200/_cluster/health?pretty

# Count documents
curl -k -u admin:YourPassword https://localhost:9200/code-search-with-vectors/_count?pretty

# Test search
curl -k -u admin:YourPassword \
  -X GET https://localhost:9200/code-search-with-vectors/_search?pretty \
  -H "Content-Type: application/json" \
  -d '{"query": {"match": {"content": "function"}}}'
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Connection Refused to OpenSearch

**Symptoms**:
```
ConnectionError: Connection refused by OpenSearch
```

**Solutions**:
1. Ensure OpenSearch is running:
   ```bash
   docker ps | grep opensearch
   ```

2. Check container logs:
   ```bash
   docker logs opensearch-node1
   ```

3. Verify port binding:
   ```bash
   curl -k -u admin:YourPassword https://localhost:9200
   ```

4. Check firewall settings (ensure port 9200 is accessible)

---

#### Issue: Authentication Failed

**Symptoms**:
```
AuthenticationException: Incorrect username or password
```

**Solutions**:
1. Verify `.env` file exists and contains password:
   ```bash
   cat .env
   ```

2. Ensure password matches in all locations:
   - `.env` file
   - `main.py` (reads from .env)
   - Ingestion scripts (read from .env)

3. Check OpenSearch logs for auth errors:
   ```bash
   docker logs opensearch-node1 | grep -i auth
   ```

---

#### Issue: No Results in Search

**Symptoms**:
- Empty search results
- Score is always 0

**Solutions**:
1. Verify data was ingested:
   ```bash
   curl -k -u admin:YourPassword \
     https://localhost:9200/code-search-with-vectors/_count
   ```

2. Check if index exists:
   ```bash
   curl -k -u admin:YourPassword \
     https://localhost:9200/_cat/indices?v
   ```

3. Verify data in index:
   ```bash
   curl -k -u admin:YourPassword \
     https://localhost:9200/code-search-with-vectors/_search?size=1&pretty
   ```

4. Re-run ingestion if needed

---

#### Issue: Embedding Model Download Fails

**Symptoms**:
```
OSError: Can't load model from HuggingFace
```

**Solutions**:
1. Check internet connection

2. Manually download model:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   ```

3. Use cached model (if previously downloaded)

4. Check disk space (model is ~90MB)

---

#### Issue: MCP Server Not Connecting from Claude

**Symptoms**:
- Claude says "Tool not available"
- MCP server not showing in Claude

**Solutions**:
1. Verify `claude_desktop_config.json` exists and is valid JSON

2. Check absolute paths in config are correct

3. Restart Claude Desktop completely (quit and reopen)

4. Check Claude Desktop logs:
   ```bash
   # MacOS
   tail -f ~/Library/Logs/Claude/mcp*.log
   ```

5. Test MCP server standalone:
   ```bash
   python main.py
   # Should show: Starting MCP Server on http://localhost:8003
   ```

---

#### Issue: Ingestion is Very Slow

**Symptoms**:
- Ingestion takes hours
- High CPU/memory usage

**Solutions**:
1. Reduce batch size in ingestion scripts:
   ```python
   if len(actions) >= 100:  # Reduced from 500
       helpers.bulk(client, actions)
   ```

2. Skip large files:
   ```python
   MAX_FILE_SIZE = 100_000  # Reduced from 500KB
   ```

3. Use text-only index (faster, no embeddings):
   ```bash
   python ingest-code/ingest-code-text-only.py
   ```

4. Filter to specific file types only

---

#### Issue: OpenSearch Container Crashes

**Symptoms**:
```
opensearch-node1 exited with code 137
```

**Solutions**:
1. Increase Docker memory limit (Settings → Resources)

2. Reduce Java heap size in `docker-compose.yml`:
   ```yaml
   - OPENSEARCH_JAVA_OPTS=-Xms256m -Xmx256m
   ```

3. Use single node instead of cluster:
   ```yaml
   # Comment out opensearch-node2 in docker-compose.yml
   ```

---

#### Issue: Index Already Exists Error

**Symptoms**:
```
resource_already_exists_exception
```

**Solutions**:
1. Delete and recreate index:
   ```bash
   curl -k -u admin:YourPassword \
     -X DELETE https://localhost:9200/code-search-with-vectors

   curl -k -u admin:YourPassword \
     -X PUT https://localhost:9200/code-search-with-vectors \
     -H "Content-Type: application/json" \
     -d @open-search/code-search-with-vectors.json
   ```

2. Or use a different index name in scripts

---

## Contributing

We welcome contributions! Here's how you can help:

### Reporting Issues

1. Check if the issue already exists
2. Provide detailed reproduction steps
3. Include logs and error messages
4. Specify your environment (OS, Python version, Docker version)

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

### Areas for Contribution

- **Language Support**: Add more programming languages
- **Search Improvements**: Enhance ranking algorithms
- **Performance**: Optimize ingestion and search speed
- **Documentation**: Improve guides and examples
- **Tests**: Add unit and integration tests
- **Features**: Multi-repository support, incremental updates, etc.

### Development Guidelines

- Follow existing code style
- Add comments for complex logic
- Update README for new features
- Test with different codebases
- Keep dependencies minimal

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

## Next Steps

Once you have your code search system running, consider these enhancements:

### 1. Index Multiple Repositories

Ingest multiple codebases to search across projects:

```bash
# Edit the ingestion script to loop through repos
repos = [
    "/path/to/repo1",
    "/path/to/repo2",
    "/path/to/repo3"
]

for repo in repos:
    walk_repo(repo)
```

### 2. Tune Search Parameters

Experiment with different configurations:

- **Increase result count**: Change `size` parameter for more results
- **Adjust boost factors**: Modify keyword vs. semantic weighting
- **Change k-NN pool size**: Larger `k` values for more semantic candidates
- **Chunk sizes**: Optimize for your codebase structure

### 3. Add Incremental Updates

Implement file-watching for automatic re-indexing:

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Watch for changes and re-index modified files
```

### 4. Monitor Performance

Use OpenSearch Dashboard to track:

- Query latency and throughput
- Index size and growth
- Search patterns and popular queries
- Resource utilization

### 5. Scale for Production

For larger deployments:

- Add more OpenSearch nodes
- Increase shard count for parallel processing
- Configure replicas for high availability
- Use dedicated hardware for embedding generation
- Implement caching for frequent queries

### 6. Extend Functionality

Add new features:

- **Multi-language support**: Add more programming languages
- **Custom analyzers**: Language-specific tokenization
- **Fuzzy matching**: Handle typos and variations
- **Syntax-aware search**: Search by AST patterns
- **Code explanations**: Use LLM to explain search results
- **Dependency tracking**: Index import/require statements

---

## References

- [OpenSearch Documentation](https://opensearch.org/docs/)
- [OpenSearch Python Client](https://opensearch-project.github.io/opensearch-py/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastMCP Documentation](https://fastmcp.dev/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/)
- [Claude Desktop](https://claude.ai/download)

---

## Acknowledgments

Built with:
- [OpenSearch](https://opensearch.org/) - Search and analytics engine
- [FastMCP](https://fastmcp.dev/) - Model Context Protocol framework
- [Sentence Transformers](https://www.sbert.net/) - Embedding generation
- [Tree-sitter](https://tree-sitter.github.io/) - Code parsing

---

## License

This project is provided as-is for code search and AI integration purposes. See [LICENSE](LICENSE) file for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/ysonawan/open-search-mcp-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ysonawan/open-search-mcp-server/discussions)
- **Documentation**: [README](README.md)

Happy code searching! 🔍
