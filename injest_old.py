"""
Injestor
--------
Injests data of the Brain File into the Qdrant Vector Database which the model uses as knowledge. 
API-KEY: UwU_pr4eAK6PFHXgeKLmlQQhgPn_us5aq5aY-1ikyEnObIxCPJ_U5Q
"""

# Imports
import csv
import os
import glob
from langchain.docstore.document import Document
from langchain.document_loaders import CSVLoader
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import json
from qdrant_client.http import models

persist_directory = "db"
source_directory = ".\\data"

# Importing config file
CONFIG = None
with open(".\\..\\config.json") as f:
    CONFIG = json.load(f)

# CSV paths and columns
prod_col = "Food Product"
# desc_col = "description"
alrg_col = "Allergens"
# link_col = "link"
ingr_col = "Ingredients"

csv_path = "dataset.csv"

# Hugging Face model for tokenization
# model_name = CONFIG["token-model-name"]
model_kwargs = {"device": 'cuda'}
encode_kwargs = {"normalize_embeddings": False}

cols_to_embed = [prod_col, alrg_col, ingr_col]
cols_to_metadata = [prod_col, alrg_col, ingr_col]

# Drafting a document with the specific data requirements for the model
docs = []
with open(csv_path, newline="", encoding='latin-1') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for i, row in enumerate(csv_reader):
        # rest of the code
        to_metadata = {col: row[col] for col in cols_to_metadata if col in row}
        values_to_embed = {k: row[k] for k in cols_to_embed if k in row}
        to_embed = ";".join(
            f"{k.strip()}=[{v.strip()}]" if v is not None else "" for k, v in values_to_embed.items())
        newDoc = Document(page_content=to_embed, metadata=to_metadata)
        docs.append(newDoc)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceBgeEmbeddings(
    model_kwargs=model_kwargs,
    # model_name = model_name,
    encode_kwargs=encode_kwargs
)

print("Embedding Model Loaded.....")

url = CONFIG["vector-db-url"]
api_key = CONFIG["vector-api-key"]
collection_name = CONFIG["vector-collection-name"]

# client = QdrantClient(
#     url=url,
#     api_key=api_key
# )

# if not client.collection_exists(collection_name):
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=models.VectorParams(
#             size=100, distance=models.Distance.COSINE
#         ),
#     )

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    api_key=api_key,
    prefer_grpc=True,
    collection_name=collection_name,
)


print("Qdrant VectorDB created...")
