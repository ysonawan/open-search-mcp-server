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
    ".cs": "c_sharp"
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
   # walk_repo("./javakiteconnect")
   # walk_repo("./duebook")
   # walk_repo("./admin-hub")
   # walk_repo("./netly")
   # walk_repo("./famvest")
   # walk_repo("./DVWA")
   # walk_repo("./git-secrets")
   # walk_repo("./juice-shop")
   pass

