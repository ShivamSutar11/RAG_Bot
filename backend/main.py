from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import uuid

# Set offline mode for local development
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from rag_engine import build_index_for_docs, answer

app = FastAPI(title="RAG_Bot API")

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory storage (as per requirements, minimal shared state but kept simple for single session)
# Key: document_id -> Value: {"filename": str, "db": ..., "bm25": ..., "chunks": ..., "type": str}
DOCUMENTS = {}

class ChatRequest(BaseModel):
    document_id: str
    question: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename
    ext = filename.split(".")[-1].lower()
    
    if ext not in ["pdf", "txt", "docx"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
        
    doc_id = str(uuid.uuid4())
    temp_path = f"/tmp/{doc_id}_{filename}"
    
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    try:
        if ext == "pdf":
            docs = PyPDFLoader(temp_path).load()
        elif ext == "txt":
            docs = TextLoader(temp_path, encoding="utf-8").load()
        else:
            raise HTTPException(status_code=400, detail="Document loader not implemented for this type")
            
        if not docs:
            raise HTTPException(status_code=400, detail="No extractable text found in document")
            
        for d in docs:
            d.metadata["source"] = filename
            
        db, bm25, chunks = build_index_for_docs(docs)
        
        DOCUMENTS[doc_id] = {
            "filename": filename,
            "db": db,
            "bm25": bm25,
            "chunks": chunks,
            "type": ext
        }
        
        return {
            "document_id": doc_id,
            "filename": filename,
            "status": "success",
            "chunk_count": len(chunks)
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/documents")
def get_documents():
    return [{"document_id": doc_id, "filename": data["filename"]} for doc_id, data in DOCUMENTS.items()]

@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    if document_id in DOCUMENTS:
        del DOCUMENTS[document_id]
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Document not found")

@app.post("/chat")
def chat(request: ChatRequest):
    if request.document_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found or index expired")
        
    doc_data = DOCUMENTS[request.document_id]
    
    try:
        # RETRIEVAL DEBUG
        print("=== RETRIEVAL DEBUG ===")
        print(f"Selected document_id: {request.document_id}")
        print(f"Selected filename: {doc_data['filename']}")
        print(f"Index object used (FAISS): {doc_data['db']}")
        print(f"Index object used (BM25): {doc_data['bm25']}")
        
        # Test what chunks are retrieved
        from rag_engine import embed_model, bm25_index
        import numpy as np
        
        # This is just for printing the debug requested by user before answering
        q_emb = embed_model.encode([request.question], convert_to_tensor=False)[0]
        d_docs = doc_data['db'].similarity_search_by_vector(q_emb, k=5)
        
        t_q = request.question.lower().split()
        b_scores = doc_data['bm25'].get_scores(t_q)
        top_n = np.argsort(b_scores)[::-1][:5]
        s_docs = [doc_data['chunks'][i] for i in top_n]
        
        all_d = d_docs + s_docs
        print("Retrieved chunk metadata:")
        for d in all_d:
            print(f" - {d.metadata}")
            print(f"   Source filename: {d.metadata.get('source')}")
        print("=======================")

        ans, best_docs = answer(request.question, 
                     custom_db=doc_data["db"], 
                     custom_bm25=doc_data["bm25"], 
                     custom_docs=doc_data["chunks"],
                     return_docs=True)
                     
        if ans == "I cannot find enough information in the uploaded document to answer this question.":
            sources = []
            source_doc = "No supporting evidence found in selected document."
        else:
            # Extract unique sources
            source_names = list(set([d.metadata.get("source") for d in best_docs if d.metadata.get("source")]))
            sources = source_names
            source_doc = ", ".join(source_names) if source_names else "Unknown"
        
        return {
            "answer": ans,
            "source_document": source_doc,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
