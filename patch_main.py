with open("backend/main.py", "r") as f:
    code = f.read()

target = """        ans = answer(request.question, 
                     custom_db=doc_data["db"], 
                     custom_bm25=doc_data["bm25"], 
                     custom_docs=doc_data["chunks"])
        
        return {
            "answer": ans,
            "source_document": doc_data["filename"],
            "sources": [] # Could optionally extract and return unique chunks here
        }"""

replacement = """        ans, best_docs = answer(request.question, 
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
        }"""

if target in code:
    code = code.replace(target, replacement)
    with open("backend/main.py", "w") as f:
        f.write(code)
    print("main.py patched")
else:
    print("target not found")
