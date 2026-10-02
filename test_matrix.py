import requests

def upload_file(path):
    with open(path, "rb") as f:
        r = requests.post("http://127.0.0.1:8000/documents/upload", files={"file": f})
        return r.json()["document_id"]

def ask(doc_id, q):
    r = requests.post("http://127.0.0.1:8000/chat", json={"document_id": doc_id, "question": q})
    return r.json()

print("Uploading documents...")
id_attn = upload_file("docs/Attention is all you need.pdf")
id_ml = upload_file("docs/machine_learning.txt")
id_cook = upload_file("docs/cooking.txt")

print(f"\n--- A) Attention -> Mitochondria ---")
res = ask(id_attn, "What is the function of mitochondria?")
print("Answer:", res['answer'])
print("Source:", res['source_document'])

print(f"\n--- B) Attention -> Scaled dot product ---")
res = ask(id_attn, "What is scaled dot-product attention?")
print("Answer:", res['answer'])
print("Source:", res['source_document'])

print(f"\n--- C) ML -> Supervised vs Unsupervised ---")
res = ask(id_ml, "What is the difference between supervised and unsupervised learning?")
print("Answer:", res['answer'])
print("Source:", res['source_document'])

print(f"\n--- D) Cooking -> Supervised vs Unsupervised ---")
res = ask(id_cook, "What is the difference between supervised and unsupervised learning?")
print("Answer:", res['answer'])
print("Source:", res['source_document'])

print(f"\n--- E) Cooking -> Bake a cake ---")
res = ask(id_cook, "How do I bake a cake?")
print("Answer:", res['answer'])
print("Source:", res['source_document'])
