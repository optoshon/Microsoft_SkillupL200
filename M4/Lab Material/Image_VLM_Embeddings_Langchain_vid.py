# Image_VLM_Embeddings_Langchain_vid.py
# ============================================================================
# MODULE 4: IMAGE INTELLIGENCE ENGINE
# ============================================================================
#
# PURPOSE:
#   Read image figures extracted by Docling (Module 1)
#   Send each image to Azure OpenAI Vision (GPT-4o)
#   Force structured JSON output (description, text, entities, etc.)
#   Embed the JSON interpretation as text for semantic search
#   Store in Chroma vector database (no hallucinations, pure structure)
#
# WHAT IT DOES:
#   1) Reads PNG/JPG/WEBP images from docling_output/*/images/
#   2) Sends each to Azure OpenAI Vision LLM (gpt-4o)
#   3) Forces JSON output (no markdown, no natural language)
#   4) Extracts: type, summary, key_text, entities, relations, numbers, warnings
#   5) Embeds the JSON interpretation using embeddings API
#   6) Stores in Chroma vector database
#
# INPUT REQUIRED:
#   - docling_output/*/images/*.png|jpg|jpeg|webp (from Module 1: Read_File_Docling.py)
#   - .env with AZURE_OPENAI_VISION_DEPLOYMENT and AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT
#
# OUTPUT CREATED:
#   - vector_db_image_vlm_chroma/ (Chroma database with image embeddings)
#   - image_vlm_manifest.json (structured interpretations of each image)
#
# RUNTIME: 60-300 seconds (slower due to Vision LLM calls; ~2-5 seconds per image)
#
# PREREQUISITE:
#   Run Module 1 (Read_File_Docling.py) BEFORE this script
#
# ============================================================================
# SETUP INSTRUCTIONS
# ============================================================================
#
# Step 1: ACTIVATE PYTHON ENVIRONMENT
#   conda activate multimodal_rag
#   # OR
#   source myenv310/bin/activate  # (macOS/Linux)
#   myenv310\Scripts\activate      # (Windows)
#
# Step 2: INSTALL DEPENDENCIES
#   If you get import errors, uncomment and run:

# !pip install python-dotenv openai langchain-core langchain-openai chromadb langchain-chroma
# OR for full requirements:
# !pip install -r requirements.txt

# Step 3: ENSURE MODULE 1 COMPLETED
#   Run: python Read_File_Docling.py
#   Verify: Check that docling_output/ folder exists with extracted images
#
# Step 4: CONFIGURE PATHS & AZURE CREDENTIALS
#   - Search for "PUT YOUR PATH HERE" below
#   - Create/update .env file with Azure OpenAI credentials
#   - Verify AZURE_OPENAI_VISION_DEPLOYMENT (must be gpt-4o or gpt-4-vision)
#   - Verify AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT (must be text-embedding-3-large or similar)
#
# Step 5: RUN THE SCRIPT (may take a while due to Vision LLM calls)
#   python Image_VLM_Embeddings_Langchain_vid.py
#
# WARNING: This module calls Vision LLM for every image, which incurs Azure costs.
#          Consider setting MAX_IMAGES to test first (e.g., MAX_IMAGES = 5)
#
# ============================================================================

from __future__ import annotations

from pathlib import Path
import os
import json
import base64
import hashlib
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import AzureOpenAI

from langchain_core.documents import Document as LCDocument
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma


# =========================
# CONFIG (edit only here)
# =========================

# PUT YOUR PATH HERE: Where Module 1 (Read_File_Docling.py) saved extracted images
DOCLING_OUTPUT_DIR = Path(
    r"C:\Users\shonr\OneDrive - Tekframeworks\Training\Microsoft\Microsoft_SkillupL200\M4\Lab Material\docling_output")


# PUT YOUR PATH HERE: Where to save Chroma vector database for images (separate from text/table DBs)

PERSIST_DIR = Path(
    r"C:\Users\shonr\OneDrive - Tekframeworks\Training\Microsoft\Microsoft_SkillupL200\M4\Lab Material\vector_image_vlm_chroma")

COLLECTION_NAME = "image_vlm_chunks_v1"

# Optional limits to avoid long runs during testing
MAX_IMAGES = None  # e.g., 5 for testing; None for all

# Vision prompt versioning (so you can regenerate later safely without conflicts)
PROMPT_VERSION = "vlm_schema_v1"

# =========================

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("image_vlm_ingest")


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:24]


def _require_env() -> Dict[str, str]:
    # PUT YOUR PATH HERE: Location of .env file with Azure OpenAI credentials
    load_dotenv(dotenv_path=Path(
        r"C:\Users\shonr\OneDrive - Tekframeworks\Secret_keys\.env"), override=False)
    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
        "AZURE_OPENAI_VISION_DEPLOYMENT",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            "Missing required .env variables (no fallbacks allowed):\n"
            + "\n".join(f"  - {k}" for k in missing)
        )
    return {k: os.environ[k] for k in required}


def find_all_images(docling_output_dir: Path) -> List[Path]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    imgs: List[Path] = []
    for ext in exts:
        imgs.extend(docling_output_dir.glob(f"*/images/{ext}"))
    imgs = sorted(set(imgs))
    if MAX_IMAGES is not None:
        imgs = imgs[:MAX_IMAGES]
    return imgs


def infer_doc_id(img_path: Path) -> str:
    return img_path.parent.parent.name


def infer_figure_id(img_path: Path) -> str:
    return img_path.stem


def to_data_url(img_path: Path) -> str:
    ext = img_path.suffix.lower()
    if ext == ".png":
        mime = "image/png"
    elif ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    else:
        raise ValueError(f"Unsupported image type for VLM: {ext}")

    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def call_vision_json(client: AzureOpenAI, deployment: str, data_url: str) -> Dict[str, Any]:
    """
    Forces JSON output. Raises if JSON cannot be parsed.
    """
    schema = """
Return ONLY valid JSON (no markdown). Schema:

{
  "type": "org_chart|flowchart|chart|table_image|screenshot|logo|photo|other",
  "summary": "1-3 sentences describing what the image shows",
  "key_text": ["important text you can read in the image"],
  "entities": [{"name": "...", "role": "..."}],
  "relations": [{"from": "...", "to": "...", "relation": "..."}],
  "numbers": [{"label": "...", "value": "...", "unit": "..."}],
  "warnings": ["uncertain/illegible parts or low-confidence claims"]
}

Rules:
- If it looks like a repeated company logo, set type="logo".
- Do NOT invent names/numbers/relationships not visible.
""".strip()

    resp = client.chat.completions.create(
        model=deployment,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a careful visual document interpreter. Output must be strict JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": schema},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )

    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("Vision model returned empty content.")

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Vision model did not return valid JSON.\nRaw:\n{content}") from e


def main():
    env = _require_env()

    img_files = find_all_images(DOCLING_OUTPUT_DIR)
    if not img_files:
        raise RuntimeError(
            f"No images found under {DOCLING_OUTPUT_DIR}.\n"
            "Expected: docling_output/<doc_id>/images/*.png (etc.)"
        )

    LOG.info("Found %d images.", len(img_files))

    # Vision client
    vision_client = AzureOpenAI(
        api_key=env["AZURE_OPENAI_API_KEY"],
        api_version=env["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
    )

    # Embeddings for storing VLM output
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

    docs: List[LCDocument] = []
    ids: List[str] = []
    manifest: List[Dict[str, Any]] = []

    for img_path in img_files:
        doc_id = infer_doc_id(img_path)
        figure_id = infer_figure_id(img_path)

        data_url = to_data_url(img_path)
        payload = call_vision_json(
            client=vision_client,
            deployment=env["AZURE_OPENAI_VISION_DEPLOYMENT"],
            data_url=data_url,
        )

        # Save sidecar JSON for audit
        sidecar = img_path.with_suffix(
            img_path.suffix + f".{PROMPT_VERSION}.json")
        sidecar.write_text(json.dumps(
            payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # Embed JSON-as-text (stable, structured)
        text_for_embedding = json.dumps(payload, ensure_ascii=False, indent=2)

        vid = _stable_id(doc_id, figure_id, PROMPT_VERSION)

        meta = {
            "doc_id": doc_id,
            "modality": "image_vlm",
            "source_image": str(img_path),
            "figure_id": figure_id,
            "vision_deployment": env["AZURE_OPENAI_VISION_DEPLOYMENT"],
            "prompt_version": PROMPT_VERSION,
            "vlm_type": payload.get("type", "unknown"),
            "sidecar_json": str(sidecar),
            "vector_id": vid,
        }

        docs.append(LCDocument(page_content=text_for_embedding, metadata=meta))
        ids.append(vid)

        manifest.append({
            "doc_id": doc_id,
            "figure_id": figure_id,
            "image": str(img_path),
            "vector_id": vid,
            "vlm_type": payload.get("type", "unknown"),
            "sidecar": str(sidecar),
        })

    if not docs:
        raise RuntimeError("No VLM outputs produced; nothing to index.")

    vectorstore.add_documents(documents=docs, ids=ids)

    (PERSIST_DIR / "image_vlm_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    LOG.info("✅ Done. VLM images indexed: %d", len(docs))
    LOG.info("Persist dir: %s", PERSIST_DIR.resolve())
    LOG.info("Collection:  %s", COLLECTION_NAME)

    # Quick sanity retrieval
    q = "organization chart reporting structure"
    hits = vectorstore.similarity_search(q, k=3)
    print("\nTop-3 VLM hits for:", q)
    for j, h in enumerate(hits, start=1):
        print(f"\n--- Hit {j} ---")
        print("meta:", {k: h.metadata.get(k)
              for k in ["doc_id", "figure_id", "vlm_type", "source_image"]})
        print("preview:", h.page_content[:300].replace("\n", " "))


if __name__ == "__main__":
    main()
