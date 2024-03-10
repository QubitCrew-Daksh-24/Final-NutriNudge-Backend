from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import json

# Importing config file
CONFIG = None
with open(".\\..\\config.json") as f:
    CONFIG = json.load(f)
print("[-] Loaded configurations")

model_kwargs = {"device": 'cuda'}
encode_kwargs = {"normalize_embeddings": False}

embeddings = HuggingFaceBgeEmbeddings(
    model_kwargs=model_kwargs,
    # model_name = model_name,
    encode_kwargs=encode_kwargs
)

url = CONFIG["vector-db-url"]
api_key = CONFIG["vector-api-key"]
collection_name = CONFIG["vector-collection-name"]

client = QdrantClient(
    url=url,
    api_key=api_key,
    # prefer_grpc=True
)

dense_vector_retriever = Qdrant(client, collection_name, embeddings)

neutral_retiever = dense_vector_retriever.as_retriever()


def searchProducts(query):
    result = neutral_retiever.get_relevant_documents(query)
    res = []
    for i in result:
        res.append(i.metadata)
    return res


if __name__ == "__main__":
    print(searchProducts("cookie"))
