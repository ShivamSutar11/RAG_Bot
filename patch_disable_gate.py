with open("rag_engine.py", "r") as f:
    code = f.read()
code = code.replace("return \"I cannot find enough information in the uploaded document to answer this question.\"", "pass # disabled gate")
with open("rag_engine.py", "w") as f:
    f.write(code)
