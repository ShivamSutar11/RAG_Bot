from openai import OpenAI
from ragas.embeddings import embedding_factory

ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
embeddings = embedding_factory('openai', model='all-minilm', client=ollama_client)
print("Worked!")
