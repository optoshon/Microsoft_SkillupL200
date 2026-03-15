# Table_Embeddings_Langchain.py
# ------------------------------------------------------------
# Purpose:
#   Load Docling-extracted tables (CSV files) from docling_output/
#   Create table-aware chunks (row groups)
#   Embed (Azure OpenAI from .env) and persist into Chroma
#
# Run:
#   python Table_Embeddings_Langchain_vid.py
#
# Requires:
#   - docling_output/ produced by Read_File_Docling.py
#   - .env with Azure OpenAI embeddings vars (same as Text_Embeddings_Langchain.py)
#
# Install:
#   pip install python-dotenv pandas langchain-core langchain-openai chromadb langchain-chroma
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import os
import re
import json
import logging
import hashlib
from typing import List, Tuple, Dict, Any

import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document as LCDocument
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma


# =========================
# CONFIG (edit only here)
# =========================

DOCLING_OUTPUT_DIR = Path("docling_output")        # same as in your scripts
PERSIST_DIR = Path("vector_db_table_chroma")       # separate DB dir for tables
COLLECTION_NAME = "table_chunks_v1"

# Table chunking strategy: group N rows per chunk, with overlap rows
ROWS_PER_CHUNK = 20
ROW_OVERLAP = 3

# Table serialization format for embedding:
#   - "markdown" is readable and tends to work well
#   - "csv" is compact but less human friendly
SERIALIZE_AS = "markdown"  # "markdown" or "csv"

# Optional limits (set to None for full)
MAX_TABLES = None          # e.g., 50
MAX_CSV_FILES = None       # e.g., 200

# =========================


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("table_ingest")


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "untitled"


def _stable_id(*parts: str) -> str:
    base = "::".join(parts)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def _require_env() -> Dict[str, str]:
    load_dotenv(dotenv_path=Path(".env"), override=False)

    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            "Missing required .env variables (no fallbacks allowed):\n"
            + "\n".join(f"  - {k}" for k in missing)
        )

    return {
        "AZURE_OPENAI_ENDPOINT": os.environ["AZURE_OPENAI_ENDPOINT"],
        "AZURE_OPENAI_API_KEY": os.environ["AZURE_OPENAI_API_KEY"],
        "AZURE_OPENAI_API_VERSION": os.environ["AZURE_OPENAI_API_VERSION"],
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT": os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
    }


def find_all_table_csvs(docling_output_dir: Path) -> List[Path]:
    # Expected: docling_output/<doc_id>/tables/table_001.csv
    csvs = sorted(docling_output_dir.glob("*/tables/*.csv"))
    if MAX_CSV_FILES is not None:
        csvs = csvs[:MAX_CSV_FILES]
    return csvs


def serialize_df(df: pd.DataFrame) -> str:
    # Keep it deterministic: no index, preserve column order
    if SERIALIZE_AS.lower() == "markdown":
        # pandas markdown output can vary slightly; keep it stable
        # Note: requires tabulate for perfect markdown in some envs, but pandas has fallback.
        return df.to_markdown(index=False)
    elif SERIALIZE_AS.lower() == "csv":
        return df.to_csv(index=False)
    else:
        raise ValueError("SERIALIZE_AS must be 'markdown' or 'csv'")


def chunk_table_rows(df: pd.DataFrame) -> List[Tuple[int, int, pd.DataFrame]]:
    """
    Returns list of (start_row_inclusive, end_row_exclusive, df_slice)
    Row indices are based on the dataframe's current row order.
    """
    if df.shape[0] == 0:
        return []

    if ROWS_PER_CHUNK <= 0:
        raise ValueError("ROWS_PER_CHUNK must be > 0")
    if ROW_OVERLAP < 0:
        raise ValueError("ROW_OVERLAP must be >= 0")
    if ROW_OVERLAP >= ROWS_PER_CHUNK:
        raise ValueError("ROW_OVERLAP must be < ROWS_PER_CHUNK")

    chunks: List[Tuple[int, int, pd.DataFrame]] = []
    step = ROWS_PER_CHUNK - ROW_OVERLAP

    start = 0
    n = df.shape[0]
    while start < n:
        end = min(start + ROWS_PER_CHUNK, n)
        df_slice = df.iloc[start:end].copy()
        chunks.append((start, end, df_slice))
        if end >= n:
            break
        start += step

    return chunks


def make_table_documents(csv_path: Path) -> Tuple[List[LCDocument], List[str]]:
    """
    Creates multiple LangChain Documents from one table CSV (row-group chunks).
    Returns: (docs, ids)
    """
    # doc_id is parent folder name: docling_output/<doc_id>/tables/...
    doc_id = csv_path.parent.parent.name
    table_id = csv_path.stem  # e.g., table_001

    df = pd.read_csv(csv_path)

    # Normalize column names (optional but makes output cleaner)
    df.columns = [str(c).strip() for c in df.columns]

    row_chunks = chunk_table_rows(df)

    docs: List[LCDocument] = []
    ids: List[str] = []

    for chunk_idx, (r0, r1, df_part) in enumerate(row_chunks, start=1):
        table_text = serialize_df(df_part)

        # Provide strong context as a "header" so embeddings know what they're seeing
        header = (
            f"TABLE CHUNK\n"
            f"doc_id: {doc_id}\n"
            f"table_id: {table_id}\n"
            f"rows: {r0}..{r1-1} (total_rows={df.shape[0]})\n"
            f"columns: {', '.join(df.columns)}\n"
            f"---\n"
        )
        content = header + table_text

        vid = _stable_id(doc_id, table_id, str(r0), str(r1), SERIALIZE_AS.lower())

        meta = {
            "doc_id": doc_id,
            "modality": "table",
            "source_csv": str(csv_path),
            "table_id": table_id,
            "chunk_index_in_table": chunk_idx,
            "row_start": int(r0),
            "row_end_exclusive": int(r1),
            "total_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "serialize_as": SERIALIZE_AS.lower(),
            "vector_id": vid,
        }

        docs.append(LCDocument(page_content=content, metadata=meta))
        ids.append(vid)

    return docs, ids


def main():
    env = _require_env()

    csv_files = find_all_table_csvs(DOCLING_OUTPUT_DIR)
    if not csv_files:
        raise RuntimeError(
            f"No table CSV files found under {DOCLING_OUTPUT_DIR}.\n"
            "Expected: docling_output/<doc_id>/tables/table_###.csv"
        )

    LOG.info("Found %d table CSV files.", len(csv_files))

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
        api_key=env["AZURE_OPENAI_API_KEY"],
        api_version=env["AZURE_OPENAI_API_VERSION"],
        azure_deployment=env["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"],
    )

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )

    total_docs = 0
    total_chunks = 0
    all_manifest: List[Dict[str, Any]] = []

    for idx, csv_path in enumerate(csv_files, start=1):
        table_docs, ids = make_table_documents(csv_path)

        if not table_docs:
            continue

        # Optional cap on number of tables processed
        if MAX_TABLES is not None and total_docs >= MAX_TABLES:
            break

        vectorstore.add_documents(documents=table_docs, ids=ids)

        total_docs += 1
        total_chunks += len(table_docs)

        all_manifest.append({
            "csv": str(csv_path),
            "doc_id": csv_path.parent.parent.name,
            "table_id": csv_path.stem,
            "chunks_created": len(table_docs),
        })

        LOG.info(
            "[%d/%d] Indexed %s -> %d chunks",
            idx, len(csv_files), csv_path.as_posix(), len(table_docs)
        )

    (PERSIST_DIR / "table_ingest_manifest.json").write_text(
        json.dumps(all_manifest, indent=2),
        encoding="utf-8"
    )

    LOG.info("✅ Done. Tables indexed: %d | Table chunks: %d", total_docs, total_chunks)
    LOG.info("Persist dir: %s", PERSIST_DIR.resolve())
    LOG.info("Collection:  %s", COLLECTION_NAME)

    # Quick sanity retrieval (table-only)
    q = "leave policy accrual maximum carry forward"
    hits = vectorstore.similarity_search(q, k=3)
    print("\nTop-3 table hits for:", q)
    for j, h in enumerate(hits, start=1):
        print(f"\n--- Hit {j} ---")
        print("meta:", {k: h.metadata.get(k) for k in ["doc_id", "table_id", "row_start", "row_end_exclusive", "source_csv"]})
        print(h.page_content[:350].replace("\n", " "))


if __name__ == "__main__":
    main()
