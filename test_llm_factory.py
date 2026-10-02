from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
llm = llm_factory("llama3.1:latest", client=ollama_client)
m = Faithfulness(llm=llm)
print("Worked!")
