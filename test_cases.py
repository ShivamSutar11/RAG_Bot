import requests

with open("docs/Attention is all you need.pdf", "rb") as f:
    resp = requests.post("http://127.0.0.1:8000/documents/upload", files={"file": f})
    data_A = resp.json()

doc_id_A = data_A["document_id"]
print(f"Doc A ID: {doc_id_A}")

q_A = "What is the difference between supervised learning and unsupervised learning?"
resp = requests.post("http://127.0.0.1:8000/chat", json={"document_id": doc_id_A, "question": q_A})
print("Case A Response:", resp.json())

q_C = "What is scaled dot-product attention?"
resp = requests.post("http://127.0.0.1:8000/chat", json={"document_id": doc_id_A, "question": q_C})
print("Case C Response:", resp.json())
