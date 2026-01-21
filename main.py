import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch

# Load environment variables from .env file
load_dotenv()

mcp = FastMCP(
    name="Code Search Agent"
)

# -----------------------------
# Embedding model (loaded once)
# -----------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

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

@mcp.prompt()
def code_search_agent():
    """Get instructions for the Code Search Agent."""
    return """
You are a code search agent powered by OpenSearch. Your primary function is to search through codebases and retrieve relevant code snippets based on plain text queries.

**Capabilities:**
- Search code repositories using natural language queries
- Perform hybrid search combining semantic understanding with keyword matching
- Filter results by repository when needed

NEVER make up code snippets. If no relevant code is found, respond with "No relevant code found."
**Response Format To User:**
Return search results with scores, repository, file path, language, line numbers, and the actual code snippet.
    """

@mcp.tool()
async def search_code(query: str, repo: str | None = None):
    """Search code snippets in OpenSearch using hybrid search.
    Args:
        query (str): The search query.
        repo (str | None): Optional repository filter.
    Returns:
        list: A list of search results with score and metadata.
    """
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Build query with optional repo filter
    query_filter = []
    if repo:
        query_filter.append({"term": {"repo": repo}})

    body = {
        "size": 10,
        "_source": {
            "excludes": ["embedding"]
        },
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": 50
                            }
                        }
                    },
                    {
                        "match": {
                            "content": {
                                "query": query,
                                "boost": 2
                            }
                        }
                    }
                ],
                "filter": query_filter
            }
        }
    }

    resp = client.search(
        index="code-search-with-vectors",
        body=body
    )

    return [
        {
            "score": hit["_score"],
            "repo": hit["_source"]["repo"],
            "path": hit["_source"]["path"],
            "language": hit["_source"]["language"],
            "start_line": hit["_source"].get("start_line"),
            "end_line": hit["_source"].get("end_line"),
            "content": hit["_source"]["content"]
        }
        for hit in resp["hits"]["hits"]
    ]


if __name__ == "__main__":
    mcp.run(transport="http", port=8003)