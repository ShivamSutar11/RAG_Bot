from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

llm = ChatOllama(model="llama3.1:latest")
wrapped_llm = LangchainLLMWrapper(llm)
m = Faithfulness(llm=wrapped_llm)
print("Worked!")
