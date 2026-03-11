
# rag_demo_stepwise_ref_obs_rerank_eval.py
# ============================================================
# STEP-BY-STEP RAG DEMO
# +  Observability
# + Re-ranking + ROUGE-L + optional RAGAS
#
# Goals:
# 1) RAG building
# 2) similarity score printing (FAISS distance; lower = more similar)
# 3) observability
# 4) reranking + evaluation
#
# Run each section independently in VSCode ("Run Cell") to see the pipeline.
# ============================================================


# %% [python]

# SETUP: Install required packages (uncomment and run if needed)
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import os
import re
import csv
import json
import time
import datetime as dt
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
!pip install torch transformers sentence-transformers langchain-community langchain langchain-text-splitters pandas python-dotenv ragas datasets


# ============================================================
# CONFIGURATION - Edit these paths / knobs
# ============================================================
CONFIG: Dict[str, Any] = {
    # Folder containing all your documents (PDF, DOCX, CSV, TXT)
    "REFERENCES_FOLDER": "references",

    # Retrieval settings
    "TOP_K": 3,
    "INITIAL_K": 10,
    "CHUNK_SIZE": 500,
    "CHUNK_OVERLAP": 50,

    # Re-ranking (DISABLED - GPT2 local only)
    "USE_RERANKER": False,
    "RERANKER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",

    # Observability
    "ENABLE_OBSERVABILITY": True,
    "OUTPUT_BASE_DIR": "outputs_stepwise",

    # Evaluation dataset
    "GOLDEN_QNA_CSV": "rag_evaluation_qna_15.csv",
    "RUN_EVALUATION": True,
    "RUN_RAGAS": True,
}
print("done")

# %%
# ============================================================
# OBSERVABILITY: Logging, artifacts (retrieval, prompt, answer, eval)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOG = logging.getLogger("rag_stepwise_observable")

OUTPUT_BASE_DIR = Path(CONFIG["OUTPUT_BASE_DIR"])
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)


def make_run_dir(base_dir: Path = OUTPUT_BASE_DIR) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False,
                    indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols = list(dict.fromkeys([k for r in rows for k in r.keys()]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


@dataclass
class CostConfig:
    chat_cost_per_1k_input: Optional[float] = None
    chat_cost_per_1k_output: Optional[float] = None
    embed_cost_per_1k_input: Optional[float] = None


COST_CFG = CostConfig(
    chat_cost_per_1k_input=None,
    chat_cost_per_1k_output=None,
    embed_cost_per_1k_input=None,
)


def estimate_cost(cfg: CostConfig, stage: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Dict[str, Any]:
    out = {
        "stage": stage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": None,
        "note": None,
        "rates_used": None,
    }
    if input_tokens is None and output_tokens is None:
        out["note"] = "Token counts not available from this call path."
        return out

    if stage == "chat":
        if cfg.chat_cost_per_1k_input is None or cfg.chat_cost_per_1k_output is None:
            out["note"] = "Chat costs not configured."
            return out
        out["rates_used"] = {
            "chat_cost_per_1k_input": cfg.chat_cost_per_1k_input,
            "chat_cost_per_1k_output": cfg.chat_cost_per_1k_output,
        }
        out["cost_usd"] = (
            (input_tokens or 0) / 1000.0 * cfg.chat_cost_per_1k_input
            + (output_tokens or 0) / 1000.0 * cfg.chat_cost_per_1k_output
        )
        return out

    if stage == "embeddings":
        if cfg.embed_cost_per_1k_input is None:
            out["note"] = "Embedding costs not configured."
            return out
        out["rates_used"] = {
            "embed_cost_per_1k_input": cfg.embed_cost_per_1k_input}
        out["cost_usd"] = (input_tokens or 0) / 1000.0 * \
            cfg.embed_cost_per_1k_input
        return out

    out["note"] = f"Unknown stage '{stage}'."
    return out


# A tiny "registry" so later cells can reuse the last run folder
_LAST_RUN_DIR: Optional[Path] = None

# %%
# ============================================================
# STEP 1: LOAD ALL DOCUMENTS FROM FOLDER  (UNCHANGED LOGIC)
# ============================================================
print("\n" + "="*60)
print("STEP 1: LOAD ALL DOCUMENTS FROM FOLDER")
print("="*60)

ref_folder = Path(CONFIG["REFERENCES_FOLDER"])
print(f"📁 Loading all documents from: {ref_folder.resolve()}")

all_files: List[Path] = []
for ext in ["*.pdf", "*.docx", "*.csv", "*.txt"]:
    all_files.extend(ref_folder.glob(ext))

if not all_files:
    raise ValueError(
        f"No documents found in {ref_folder}. Put your PDFs/DOCXs/CSVs/TXTs inside it.")

print(f"📄 Found {len(all_files)} file(s):")
for f in all_files:
    print(f"   - {f.name}")

documents = []
for file_path in all_files:
    print(f"\n🔄 Loading: {file_path.name}")

    if file_path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif file_path.suffix.lower() == ".docx":
        loader = Docx2txtLoader(str(file_path))
    elif file_path.suffix.lower() == ".csv":
        loader = CSVLoader(str(file_path))
    elif file_path.suffix.lower() == ".txt":
        loader = TextLoader(str(file_path))
    else:
        print(f"   ⚠️ Skipping unsupported file type: {file_path.suffix}")
        continue

    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file_path.name
    documents.extend(docs)

print(f"\n✅ Loaded {len(documents)} document(s) total")
print("\n📋 First document preview:")
print("-" * 60)
print(documents[0].page_content[:500])
print("-" * 60)
print(f"\n📊 Source: {documents[0].metadata.get('source', 'Unknown')}")

# %%
# ============================================================
# STEP 2: SPLIT INTO CHUNKS
# ============================================================
print("\n" + "="*60)
print("STEP 2: SPLIT INTO CHUNKS")
print("="*60)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CONFIG["CHUNK_SIZE"],
    chunk_overlap=CONFIG["CHUNK_OVERLAP"],
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = text_splitter.split_documents(documents)

print(f"✅ Created {len(chunks)} chunks from {len(all_files)} file(s)")
print(f"📏 Chunk size: {CONFIG['CHUNK_SIZE']} characters")
print(f"🔗 Chunk overlap: {CONFIG['CHUNK_OVERLAP']} characters")

print("\n📝 Sample chunks:")
for i, chunk in enumerate(chunks[:3], 1):
    print(
        f"\n--- Chunk {i} (from {chunk.metadata.get('source', 'Unknown')}) ({len(chunk.page_content)} chars) ---")
    print(chunk.page_content[:200] +
          "..." if len(chunk.page_content) > 200 else chunk.page_content)

# %%
# ============================================================
# STEP 3: GENERATE EMBEDDINGS & CREATE VECTOR DATABASE
# ============================================================
print("\n" + "="*60)
print("STEP 3: GENERATE EMBEDDINGS & CREATE VECTOR DATABASE")
print("="*60)

print("🔄 Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class SentenceTransformerEmbeddings:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, text: str):
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text: str):
        return self.model.encode([text])[0].tolist()

    def get_sentence_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


embeddings = SentenceTransformerEmbeddings(model)

print("🔄 Generating embeddings and building FAISS index...")
vector_store = FAISS.from_documents(chunks, embeddings)

print("✅ Vector database created!")
print(f"📊 Total vectors: {vector_store.index.ntotal}")
print(
    f"📐 Embedding dimension: {embeddings.get_sentence_embedding_dimension()}")

print(f"\n🧪 Test embedding for '{test_text}':")
print(f"   Vector length: {len(test_embedding)}")
print(f"   First 5 values: {test_embedding[:5]}")

# %%
# ============================================================
# STEP 4: RETRIEVE RELEVANT CHUNKS
# ============================================================
print("\n" + "="*60)
print("STEP 4: RETRIEVE RELEVANT CHUNKS")
print("="*60)

query = "Within how many working days of the Date of Joining must an employee be enrolled under EPF and ESI?"
print(f"🔍 Query: {query}")

t0_total = time.perf_counter()
run_dir = make_run_dir() if CONFIG["ENABLE_OBSERVABILITY"] else None
_LAST_RUN_DIR = run_dir

if run_dir:
    LOG.info("Run dir: %s", run_dir.as_posix())

t_retr0 = time.perf_counter()

retrieved_with_scores = vector_store.similarity_search_with_score(
    query,
    k=int(CONFIG["INITIAL_K"] if CONFIG["USE_RERANKER"] else CONFIG["TOP_K"]),
)
t_retr1 = time.perf_counter()

cand_docs = [d for (d, _s) in retrieved_with_scores]
cand_scores = [float(_s) for (_d, _s) in retrieved_with_scores]

print(f"\n🔄 Searching for top {len(cand_docs)} most similar chunks...")
print(f"✅ Retrieved {len(cand_docs)} relevant chunks (candidates):")
for i, doc in enumerate(cand_docs[: min(3, len(cand_docs))], 1):
    print(
        f"\n--- Retrieved Chunk {i} (Source: {doc.metadata.get('source', 'Unknown')}) ---")
    print(f"Content: {doc.page_content[:300]}...")
    print("-" * 60)

print(f"\n📊 Similarity Scores (lower = more similar):")
for i, (doc, score) in enumerate(retrieved_with_scores[: int(CONFIG["TOP_K"])], 1):
    source = doc.metadata.get("source", "Unknown")
    print(f"   Chunk {i} [{source}]: {float(score):.4f}")

retrieved_docs = cand_docs[: int(CONFIG["TOP_K"])]
if run_dir:
    retrieval_rows: List[Dict[str, Any]] = []
    for rank, ((doc, score)) in enumerate(retrieved_with_scores, start=1):
        md = doc.metadata or {}
        retrieval_rows.append({
            "rank": rank,
            "source": md.get("source", "Unknown"),
            "faiss_distance": float(score),
            "rerank_score": md.get("rerank_score"),
            "preview": (doc.page_content or "")[:220].replace("\n", " "),
        })

    write_csv(run_dir / "retrieval_candidates.csv", retrieval_rows)

    dump_lines = []
    for rank, (doc, score) in enumerate(retrieved_with_scores, start=1):
        md = doc.metadata or {}
        dump_lines.append(
            f"[{rank}] source={md.get('source', 'Unknown')} | faiss_distance={float(score):.6f} | rerank={md.get('rerank_score')}\n"
            + (doc.page_content or "")
        )
    write_text(run_dir / "retrieval_candidates.txt",
               "\n\n" + ("-"*80 + "\n\n").join(dump_lines))

    LOG.info("Retrieved candidates=%d in %.3fs", len(
        retrieved_with_scores), (t_retr1 - t_retr0))
    if CONFIG["USE_RERANKER"]:
        LOG.info("Reranked in %.3fs", (t_rr1 - t_rr0))

# %%
# ============================================================
# STEP 5: GENERATE ANSWER (GPT-2 LOCAL)
# ============================================================
print("\n" + "="*60)
print("STEP 5: GENERATE ANSWER")
print("="*60)

model_name = "gpt2"  # or "gpt2-medium" if you have more RAM
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

context_text = "\n\n".join([
    f"[Context {i+1}]\n{doc.page_content}"
    for i, doc in enumerate(retrieved_docs)
])

rag_prompt = f"""
Answer the question using ONLY the context below.
If the context does not contain the answer, say:
"I don't know based on the provided context."

Context:
{context_text}

Question:
{query}

Answer:
"""

print("📝 RAG Prompt created")
print("-" * 60)
print(rag_prompt)

print("\n🔄 Generating answer from local GPT-2...")

t_llm0 = time.perf_counter()
out = generator(
    rag_prompt,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
t_llm1 = time.perf_counter()

final_answer = out[0]["generated_text"][len(rag_prompt):].strip()

print("\n" + "="*60)
print("✨ FINAL ANSWER")
print("="*60)
print(f"Question: {query}")
print(f"\nAnswer: {final_answer}")
print("="*60)

print("\n📌 Sources used:")
for d in retrieved_docs:
    print(" -", d.metadata.get("source", "Unknown"))

if _LAST_RUN_DIR:
    write_text(_LAST_RUN_DIR / "rag_prompt.txt", rag_prompt)
    write_text(_LAST_RUN_DIR / "final_answer.txt", final_answer)
    LOG.info("Generated answer in %.3fs", (t_llm1 - t_llm0))

# %%
# ============================================================
# STEP 6: EVALUATION (ROUGE-L)
# ============================================================
print("\n" + "="*60)
print("STEP 6: EVALUATION (ROUGE-L)")
print("="*60)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _lcs_length(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def rouge_l_f1(pred: str, ref: str) -> float:
    pred_toks = _tokenize(pred)
    ref_toks = _tokenize(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    lcs = _lcs_length(pred_toks, ref_toks)
    prec = lcs / len(pred_toks)
    rec = lcs / len(ref_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def retrieve_with_scores(q: str, k: int) -> List[Tuple[Any, float]]:
    # We intentionally use with_score so we can log distances deterministically.
    return vector_store.similarity_search_with_score(q, k=k)


def run_single_query(q: str) -> Tuple[str, List[Any], List[Tuple[Any, float]]]:
    # candidates
    hits_scored = retrieve_with_scores(q, k=int(CONFIG["TOP_K"]))
    docs = [d for (d, _s) in hits_scored]
    final_docs = docs

    # Build context
    ctx = "\n\n".join(
        [f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}" for d in final_docs])
    prmpt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say: "I don't have enough information in the provided documents."

Question:
{q}

Context:
{ctx}

Answer:
"""

    # Try to use GPT-4o-mini if OPENAI_API_KEY is available, otherwise fall back to GPT-2
    try:
        if os.getenv("OPENAI_API_KEY"):
            from langchain_openai import ChatOpenAI
            openai_llm = ChatOpenAI(
                model="gpt-4o-mini", temperature=0, max_tokens=300)
            pred = openai_llm.invoke(prmpt).content
        else:
            o = generator(
                prmpt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            pred = o[0]["generated_text"][len(prmpt):].strip()
    except Exception:
        # Fallback to local GPT-2
        o = generator(
            prmpt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        pred = o[0]["generated_text"][len(prmpt):].strip()

    return pred, final_docs, hits_scored


if not CONFIG["RUN_EVALUATION"]:
    print("⏭️ Evaluation disabled in CONFIG['RUN_EVALUATION'].")
else:
    import pandas as pd

    golden_path = Path(CONFIG["GOLDEN_QNA_CSV"])
    if not golden_path.exists():
        raise FileNotFoundError(
            f"Golden CSV not found: {golden_path.resolve()}\n"
            f"Tip: Put it next to this script or update CONFIG['GOLDEN_QNA_CSV']."
        )

    golden_df = pd.read_csv(golden_path)
    required_cols = {"question", "answer"}
    if not required_cols.issubset(set(golden_df.columns)):
        raise ValueError(
            f"Golden CSV must contain columns {sorted(required_cols)}. Found: {list(golden_df.columns)}")

    eval_rows: List[Dict[str, Any]] = []
    t_eval0 = time.perf_counter()

    eval_run_dir = make_run_dir() if CONFIG["ENABLE_OBSERVABILITY"] else None
    if eval_run_dir:
        LOG.info("Eval run dir: %s", eval_run_dir.as_posix())

    for idx, r in golden_df.iterrows():
        q = str(r["question"])
        gt = str(r["answer"])

        pred, final_docs, hits_scored = run_single_query(q)
        rl = rouge_l_f1(pred, gt)

        sources = [d.metadata.get("source", "Unknown") for d in final_docs]
        contexts = [d.page_content for d in final_docs]

        eval_rows.append({
            "idx": idx,
            "question": q,
            "ground_truth": gt,
            "prediction": pred,
            "rougeL_f1": rl,
            "sources": " | ".join(sources),
            "contexts_joined": "\n\n---\n\n".join(contexts),
        })

        print(f"✅ [{idx+1}/{len(golden_df)}] ROUGE-L(F1)={rl:.3f} | {q[:70]}...")

        # Optional: per-item observability dump
        if eval_run_dir:
            per_q_dir = eval_run_dir / f"q_{idx:03d}"
            per_q_dir.mkdir(parents=True, exist_ok=False)

            # Save candidate distances (FAISS)
            cand_rows = []
            for rank, (doc, dist) in enumerate(hits_scored, start=1):
                cand_rows.append({
                    "rank": rank,
                    "source": (doc.metadata or {}).get("source", "Unknown"),
                    "faiss_distance": float(dist),
                    "rerank_score": (doc.metadata or {}).get("rerank_score"),
                    "preview": (doc.page_content or "")[:220].replace("\n", " "),
                })
            write_csv(per_q_dir / "candidates.csv", cand_rows)

            # Save final context
            write_text(per_q_dir / "final_context.txt",
                       "\n\n---\n\n".join(contexts))

            # Save prediction + gt
            write_json(per_q_dir / "result.json", {
                "question": q,
                "ground_truth": gt,
                "prediction": pred,
                "rougeL_f1": rl,
                "sources": sources,
            })

    t_eval1 = time.perf_counter()
    results_df = pd.DataFrame(eval_rows)

    print("\n" + "-" * 60)
    print("ROUGE-L summary")
    print("-" * 60)
    print(results_df["rougeL_f1"].describe())
    print(f"\n⏱️ Evaluation runtime: {t_eval1 - t_eval0:.1f}s")

    out_csv = Path("rag_eval_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n💾 Saved evaluation results to: {out_csv.resolve()}")

    if eval_run_dir:
        write_csv(eval_run_dir / "rag_eval_results.csv",
                  results_df.to_dict(orient="records"))

# %%
# ============================================================
# STEP 7: RAGAS EVALUATION (Optional - requires OpenAI API key)
# ============================================================
print("\n" + "="*60)
print("STEP 7: RAGAS EVALUATION (Optional)")
print("="*60)

# Load .env file for API keys
try:
    from dotenv import load_dotenv
    load_dotenv(r"C:\Users\shonr\OneDrive - Tekframeworks\Secret_keys\.env")
    openai_key = os.getenv("OPENAI_API_KEY")
    print("✅ .env file loaded (if present)")
    if openai_key:
        print("✅ OPENAI_API_KEY found in .env")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

if not CONFIG["RUN_RAGAS"]:
    print("\n⏭️ RAGAS disabled in CONFIG['RUN_RAGAS'].")
else:
    print("\n" + "-" * 60)
    print("RAGAS Evaluation")
    print("-" * 60)

    try:
        from ragas import evaluate
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        print("✅ Configuring RAGAS with GPT-4o-mini...")

        # Configure RAGAS to use GPT-4o-mini for evaluation
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    except Exception as e:
        print(f"⚠️ RAGAS not available ({e}). Skipping RAGAS metrics.")
        print(f"Tip: Install with: pip install ragas datasets langchain-openai")
    else:
        # RAGAS expects: question, answer (model response), contexts (list[str]), ground_truth
        ds = Dataset.from_list([
            {
                "question": row["question"],
                "answer": row["prediction"],
                "contexts": row["contexts_joined"].split("\n\n---\n\n"),
                "ground_truth": row["ground_truth"],
            }
            for row in eval_rows
        ])

        try:
            print("🔄 Running RAGAS with GPT-4o-mini...")

            # Evaluate with LLM and embeddings passed directly
            ragas_report = evaluate(
                ds,
                llm=llm,
                embeddings=embeddings,
            )
            ragas_df = ragas_report.to_pandas()
            print("\n✅ RAGAS per-row metrics:")
            print(ragas_df)
            print("\n📈 RAGAS averages:")
            print(ragas_df.mean(numeric_only=True))

            # Merge RAGAS results with ROUGE results
            merged = results_df.merge(
                ragas_df, left_on="idx", right_index=True, how="left")
            results_df = merged
            print(f"\n✅ Merged RAGAS metrics with ROUGE results")
        except Exception as e:
            print(
                "⚠️ RAGAS evaluation failed (likely missing eval LLM configuration).\n"
                f"Error: {e}\n"
                f"Tip: Make sure OPENAI_API_KEY is set in .env file"
            )

    # Save final results
    out_csv = Path("rag_eval_results_with_ragas.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n💾 Saved RAGAS evaluation results to: {out_csv.resolve()}")

# %%
# ============================================================
# STEP 8: SUMMARY
# ============================================================
print("\n" + "="*60)
print("📊 RAG PIPELINE SUMMARY")
print("="*60)
print(f"1. Loaded: {len(all_files)} file(s) → {len(documents)} document(s)")
print(f"2. Split into: {len(chunks)} chunks")
print(f"3. Created vector DB with: {vector_store.index.ntotal} embeddings")
print(f"4. Retrieved: {CONFIG['TOP_K']} relevant chunks (final)")
print(f"5. Generated grounded answer ✅")
print(f"6. ROUGE-L evaluation completed ✅")
if CONFIG["RUN_RAGAS"]:
    print(f"7. RAGAS evaluation completed (if available) ✅")
if _LAST_RUN_DIR:
    print(f"\n📂 Observability artifacts: {_LAST_RUN_DIR.as_posix()}")
print("="*60)

print("\n🎉 RAG Demo Complete!")
print("\n💡 To try a different question, change the 'query' variable and re-run from there.")
