import sys

with open("rag_engine.py", "r") as f:
    lines = f.readlines()

new_imports = """from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, PyPDFDirectoryLoader
from langchain.embeddings.base import Embeddings
from sentence_transformers.cross_encoder import CrossEncoder
import os
import torch
import numpy as np
from rank_bm25 import BM25Okapi
"""

# Replace lines 3-10
new_lines = lines[:2] + [new_imports] + lines[10:]

with open("rag_engine.py", "w") as f:
    f.writelines(new_lines)
