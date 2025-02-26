from openai import OpenAI
from get_API_key import *

# openAI Embedding

api_key = get_API_key()

client = OpenAI(
    api_key=api_key,
)
GPT_MODEL = "gpt-4o-mini"


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
    return embedding
