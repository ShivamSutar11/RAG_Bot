# rag_engine.py

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.base import Embeddings
import os
import torch

# ------------------------------------------------
# 1) LOAD QWEN 0.5B MODEL
# ------------------------------------------------

print("⚡ Loading Qwen2.5-0.5B-Instruct model...")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True
)

# Prevent model from echoing system/user tags
model.generation_config.stop_strings = ["<|im_end|>", "<|im_start|>"]

print("✅ Model loaded.")


# ------------------------------------------------
# 2) LOAD EMBEDDINGS
# ------------------------------------------------

print("🔍 Loading MiniLM embeddings...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ------------------------------------------------
# 3) LOAD DOCUMENTS
# ------------------------------------------------

def load_documents():
    docs = []

    if not os.path.exists("./docs"):
        raise Exception("❗ Create a folder named 'docs' and put PDFs/TXTs inside it.")

    for f in os.listdir("./docs"):
        path = os.path.join("./docs", f)

        if f.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())

        elif f.endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())

    print(f"📄 Loaded {len(docs)} documents.")
    return docs


# ------------------------------------------------
# 4) BUILD FAISS INDEX
# ------------------------------------------------

def build_faiss():
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]

    class MiniEmb(Embeddings):
        def embed_documents(self, docs):
            return embed_model.encode(docs, convert_to_tensor=False)

        def embed_query(self, text):
            return embed_model.encode([text], convert_to_tensor=False)[0]

    db = FAISS.from_texts(texts, MiniEmb())
    print("✅ FAISS index built.")
    return db


db = build_faiss()


# ------------------------------------------------
# 5) CLEAN OUTPUT — ***THIS FIXES YOUR ISSUE***
# ------------------------------------------------

def clean_output(text, question):
    # Extract only the assistant part
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]

    remove_words = ["<|im_end|>", "<|im_start|>", "system",
                    "user", "assistant"]

    for w in remove_words:
        text = text.replace(w, "")

    # Remove repeated question
    text = text.replace(f"Question: {question}", "")

    return text.strip()


# ------------------------------------------------
# 6) RAG ANSWERING FUNCTION
# ------------------------------------------------

def rag_answer(question, k=3):

    query_emb = embed_model.encode([question], convert_to_tensor=False)[0]
    docs = db.similarity_search_by_vector(query_emb, k=k)

    context = "\n\n".join([d.page_content for d in docs])

    system_prompt = (
        "You are a STRICT RAG ASSISTANT.\n"
        "You MUST answer ONLY using the given context.\n"
        "If the answer is not fully in the context, reply exactly:\n"
        "'I cannot find that information in the context.'\n"
        "Do NOT use outside knowledge.\n"
    )

    prompt = (
        "<|im_start|>system\n"
        f"{system_prompt}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    output = model.generate(
        input_ids,
        max_new_tokens=150,
        temperature=0.0,
        do_sample=False,
    )

    raw = tokenizer.decode(output[0], skip_special_tokens=False)

    cleaned = clean_output(raw, question)

    return cleaned


# ------------------------------------------------
# 7) EXPOSE FOR UI
# ------------------------------------------------
def answer(query):
    return rag_answer(query)
