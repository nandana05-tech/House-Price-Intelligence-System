"""
Embed static markdown knowledge documents into pgvector.
Run once — and re-run only when knowledge/ files are edited:
docker exec -it hpi_api python scripts/ingest_knowledge.py
"""
import os
import re
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_postgres.vectorstores import PGVector
from rag.embedder import get_embeddings
from langchain_text_splitters import MarkdownTextSplitter

_DATABASE_URL = os.getenv("DATABASE_URL", "")
# langchain-postgres PGVector requires the psycopg3 driver prefix.
PGVECTOR_URL = re.sub(r"^postgresql(\+psycopg2)?", "postgresql+psycopg", _DATABASE_URL)

KNOWLEDGE_DIR = Path("knowledge")
splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)

doc_type_map = {
    "area_profiles": "area_profile",
    "market_rules": "market_rule",
    "faqs": "faq",
}

all_docs = []
for folder, doc_type in doc_type_map.items():
    for md_file in (KNOWLEDGE_DIR / folder).glob("*.md"):
        loader = TextLoader(str(md_file), encoding="utf-8")
        raw_docs = loader.load()
        chunks = splitter.split_documents(raw_docs)
        for chunk in chunks:
            chunk.metadata.update({
                "doc_type": doc_type,
                "title": md_file.stem.replace("_", " ").title(),
                "source": str(md_file),
            })
        all_docs.extend(chunks)

vectorstore = PGVector(
    embeddings=get_embeddings(),
    collection_name="knowledge_base",
    connection=PGVECTOR_URL,
    use_jsonb=True,
)
vectorstore.add_documents(all_docs)
print(f"Ingested {len(all_docs)} knowledge chunks.")