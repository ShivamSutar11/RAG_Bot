import os
with open("evals/run_eval.py", "r") as f:
    code = f.read()

# Add sys.path.append for backend before importing rag_engine
import_stmt = "from rag_engine import answer"
sys_path_stmt = """import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from rag_engine import answer"""

code = code.replace(import_stmt, sys_path_stmt)

with open("evals/run_eval.py", "w") as f:
    f.write(code)
print("run_eval patched")
