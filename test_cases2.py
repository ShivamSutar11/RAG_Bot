import requests

with open("docs/Attention is all you need.pdf", "rb") as f:
    resp1 = requests.post("http://127.0.0.1:8000/documents/upload", files={"file": f})
    doc_id_attn = resp1.json()["document_id"]
print(f"Attention Doc ID: {doc_id_attn}")

# Test Case A: Attention -> Supervised learning (expected to refuse)
q_A = "What is the difference between supervised learning and unsupervised learning?"
resp_A = requests.post("http://127.0.0.1:8000/chat", json={"document_id": doc_id_attn, "question": q_A})
print("Case A Response:", resp_A.json())

# Test Case C: Attention -> Scaled dot product
q_C = "What is scaled dot-product attention?"
resp_C = requests.post("http://127.0.0.1:8000/chat", json={"document_id": doc_id_attn, "question": q_C})
print("Case C Response:", resp_C.json())

