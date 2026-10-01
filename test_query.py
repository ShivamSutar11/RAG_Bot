import traceback
from rag_engine import answer

try:
    print("Testing query...")
    res = answer("What is this document about?")
    print("Response:", res)
except Exception as e:
    print("ERROR CAUGHT:")
    traceback.print_exc()
