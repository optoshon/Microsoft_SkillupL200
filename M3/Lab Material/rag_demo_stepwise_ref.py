# rag_demo_stepwise.py
# ============================================================
# STEP-BY-STEP RAG DEMO
# Run each section independently in VSCode to see RAG in action
# ============================================================


# %% [python]

# SETUP: Uncomment and run the line below to install required packages
# pip install torch transformers sentence-transformers langchain-community langchain langchain-text-splitters python-dotenv

from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ============================================================
# MODEL CONFIGURATION
# ============================================================
# CURRENT: Using GPT-2 (Local/Open-source)
# - Runs locally on your CPU/GPU
# - No API key required
# - No external API calls
#
# TO USE OTHER MODELS (GPT 5.2, GPT-4, Claude, etc.):
#
# 1. Create a .env file in the project root with your API keys:
#    OPENAI_API_KEY=sk-your-api-key-here
#    ANTHROPIC_API_KEY=your-api-key-here
#    HF_API_TOKEN=hf_your-token-here
#
# 2. Load the .env file at the start of your script:
#    from dotenv import load_dotenv
#    import os
#    load_dotenv()
#    api_key = os.getenv("OPENAI_API_KEY")
#
# 3. Update the model_name in Step 5 and Step 7
#    model_name = "gpt-4"  # or "gpt-3.5-turbo", "claude-3-sonnet", etc.
#
# 4. Install required libraries:
#    pip install python-dotenv openai anthropic
# ============================================================

CONFIG = {
    # Folder containing all your documents (PDF, DOCX, CSV, TXT)
    "REFERENCES_FOLDER": "references",  # Put all your documents here!


    # Retrieval settings
    "TOP_K": 3,
    "CHUNK_SIZE": 500,
    "CHUNK_OVERLAP": 50,
}
print("done")

# %%
# ============================================================
# STEP 1: LOAD ALL DOCUMENTS FROM FOLDER
# ============================================================
print("\n" + "="*60)
print("STEP 1: LOAD ALL DOCUMENTS FROM FOLDER")
print("="*60)

ref_folder = Path(CONFIG["REFERENCES_FOLDER"])
print(f"📁 Loading all documents from: {ref_folder}")

all_files = []
for ext in ['*.pdf', '*.docx', '*.csv', '*.txt']:
    all_files.extend(ref_folder.glob(ext))

if not all_files:
    raise ValueError(f"No documents found in {ref_folder}")

print(f"📄 Found {len(all_files)} file(s):")
for f in all_files:
    print(f"   - {f.name}")

# Load all documents
documents = []
for file_path in all_files:
    print(f"\n🔄 Loading: {file_path.name}")

    if file_path.suffix.lower() == '.pdf':
        loader = PyPDFLoader(str(file_path))
    elif file_path.suffix.lower() == '.docx':
        loader = Docx2txtLoader(str(file_path))
    elif file_path.suffix.lower() == '.csv':
        loader = CSVLoader(str(file_path))
    elif file_path.suffix.lower() == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(file_path))
    else:
        print(f"   ⚠️ Skipping unsupported file type: {file_path.suffix}")
        continue

    docs = loader.load()
    for doc in docs:
        doc.metadata['source'] = file_path.name
    documents.extend(docs)

print(f"\n✅ Loaded {len(documents)} document(s) total")
print(f"\n📋 First document preview:")
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
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)

print(f"✅ Created {len(chunks)} chunks from {len(all_files)} file(s)")
print(f"📏 Chunk size: {CONFIG['CHUNK_SIZE']} characters")
print(f"🔗 Chunk overlap: {CONFIG['CHUNK_OVERLAP']} characters")

print(f"\n📝 Sample chunks:")
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
    def __init__(self, model):
        self.model = model

    def __call__(self, text: str):
        return self.embed_query(text)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

    def get_sentence_embedding_dimension(self):
        return self.model.get_sentence_embedding_dimension()


embeddings = SentenceTransformerEmbeddings(model)

print("🔄 Generating embeddings and building FAISS index...")
vector_store = FAISS.from_documents(chunks, embeddings)

print(f"✅ Vector database created!")
print(f"📊 Total vectors: {vector_store.index.ntotal}")
print(
    f"📐 Embedding dimension: {embeddings.get_sentence_embedding_dimension()}")


# Test embedding
test_text = "What is the sick leave policy?"
test_embedding = embeddings.embed_query(test_text)
print(f"\n🧪 Test embedding for '{test_text}':")
print(f"   Vector length: {len(test_embedding)}")
print(f"   First 5 values: {test_embedding[:5]}")

# %%
# ============================================================
# STEP 4: RETRIEVE RELEVANT CHUNKS (QUERY)
# ============================================================
print("\n" + "="*60)
print("STEP 4: RETRIEVE RELEVANT CHUNKS")
print("="*60)

# Your demo query - CHANGE THIS!
query = "What is the sick leave policy?"
print(f"🔍 Query: {query}")

print(f"\n🔄 Searching for top {CONFIG['TOP_K']} most similar chunks...")
retrieved_with_scores = vector_store.similarity_search_with_score(
    query, k=CONFIG["TOP_K"])
retrieved_docs = [doc for doc, _ in retrieved_with_scores]

print(f"✅ Retrieved {len(retrieved_docs)} relevant chunks:")
for i, (doc, score) in enumerate(retrieved_with_scores, 1):
    print(
        f"\n--- Retrieved Chunk {i} (Source: {doc.metadata.get('source', 'Unknown')}) | Score: {score:.4f} ---")
    print(f"Content: {doc.page_content[:300]}...")
    print("-" * 60)
for i, (doc, score) in enumerate(retrieved_with_scores, 1):
    source = doc.metadata.get('source', 'Unknown')
    print(f"   Chunk {i} [{source}]: {score:.4f}")

# %%
# ============================================================
# STEP 5: GENERATE ANSWER (LOCAL GPT-2) - NO RAG
# ============================================================
print("\n" + "="*60)
print("STEP 5: GENERATE ANSWER (LOCAL GPT-2 - NO RAG)")
print("="*60)

query = "Within how many working days of the Date of Joining must an employee be enrolled under EPF and ESI?"

# MODEL: GPT-2 LOCAL (runs locally, no API)
model_name = "gpt2"  # or "gpt2-medium" if you have more RAM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

prompt = query  # No context - direct query

out = generator(
    prompt,
    max_new_tokens=150,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

answer = out[0]["generated_text"][len(prompt):]

print("\n✅ Answer (GPT-2, local - no context):")
print(f"Query: {query}")
print(f"Answer: {answer}")


# %%
# ============================================================
# STEP 6: RETRIEVE RELEVANT CHUNKS & BUILD CONTEXT
# ============================================================
print("\n" + "="*60)
print("STEP 6: RETRIEVE RELEVANT CHUNKS & BUILD CONTEXT")
print("="*60)

query = "Within how many working days of the Date of Joining must an employee be enrolled under EPF and ESI?"
print(f"🔍 Query: {query}")

print(f"\n🔄 Searching for top {CONFIG['TOP_K']} most similar chunks...")
retrieved_docs = vector_store.similarity_search(query, k=CONFIG["TOP_K"])

print(f"✅ Retrieved {len(retrieved_docs)} relevant chunks:")
for i, doc in enumerate(retrieved_docs, 1):
    print(
        f"\n--- Chunk {i} (Source: {doc.metadata.get('source', 'Unknown')}) ---")
    print(f"{doc.page_content[:300]}...")

context_text = "\n\n".join([
    f"[Context {i+1}]\n{doc.page_content}"
    for i, doc in enumerate(retrieved_docs)
])


# %%
# ============================================================
# STEP 7: GENERATE GROUNDED RESPONSE (GPT-2 LOCAL WITH RAG)
# ============================================================
print("\n" + "="*60)
print("STEP 7: GENERATE GROUNDED RESPONSE (GPT-2 LOCAL WITH RAG)")
print("="*60)

# Create RAG prompt with context
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

print("🔄 Generating answer from local GPT-2 with RAG context...")

out = generator(
    rag_prompt,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

final_answer = out[0]["generated_text"][len(rag_prompt):].strip()

print("\n" + "="*60)
print("✨ FINAL RAG ANSWER")
print("="*60)
print(f"Question: {query}")
print(f"\nAnswer: {final_answer}")
print("="*60)

print("\n📌 Sources used:")
for d in retrieved_docs:
    print(" -", d.metadata.get("source", "Unknown"))

# %%
# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("📊 RAG PIPELINE SUMMARY")
print("="*60)
print(f"1. Loaded: {len(all_files)} file(s) → {len(documents)} document(s)")
print(f"2. Split into: {len(chunks)} chunks")
print(f"3. Created vector DB with: {vector_store.index.ntotal} embeddings")
print(f"4. Retrieved: {CONFIG['TOP_K']} relevant chunks")
# print(f"5. LLM: {Path(CONFIG['MODEL_PATH']).name}")
print(f"6. Generated grounded answer ✅")
print("="*60)

print("\n🎉 RAG Demo Complete!")
print("\n💡 To try a different question, change the 'query' variable and re-run from there.")

# %%
