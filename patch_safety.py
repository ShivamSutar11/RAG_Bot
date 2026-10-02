import os
with open("backend/rag_engine.py", "r") as f:
    code = f.read()

target = """    candidates = list(unique_docs.values())"""
replacement = """    candidates = list(unique_docs.values())
    
    # Hard safety check for document isolation
    if custom_docs:
        expected_source = custom_docs[0].metadata.get("source")
        for doc in candidates:
            if doc.metadata.get("source") != expected_source:
                print(f"HARD SAFETY ERROR: Retrieved chunk from {doc.metadata.get('source')}, expected {expected_source}")
                return "I cannot find enough information in the uploaded document to answer this question."
"""

if target in code:
    code = code.replace(target, replacement)
    with open("backend/rag_engine.py", "w") as f:
        f.write(code)
    print("Patched safety check.")
else:
    print("Could not find target string.")
