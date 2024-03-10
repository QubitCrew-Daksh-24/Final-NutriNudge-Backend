from qdrant_client import QdrantClient
from qdrant_client.http import models
import json


# Importing config file
CONFIG = None
with open(".\\..\\config.json") as f:
    CONFIG = json.load(f)
print("[-] Loaded configurations")

client = QdrantClient(url=CONFIG["vector-db-url"],
                      api_key=CONFIG["vector-api-key"])


def searchProducts(query, allergens):
    res = client.search(
        collection_name=CONFIG["vector-collection-name"],
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="Food Product",
                    match=models.MatchValue(
                        value=query,
                    ),
                ),
                models.FieldCondition(
                    key="Ingredients",
                    match=models.MatchAny(
                        any=allergens,
                        must_not=True  # This is the key change, indicating that the ingredients must not match any of the allergens
                    )
                )
            ]
        ),
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        query_vector=[0.2, 0.1, 0.9, 0.7],
        # limit=3,
        with_payload=["Food Product", "Ingredients", "Allergens"],
    )
    return res


def search_non_allergic_products(query, allergens):
    out = client.scroll(
        collection_name=CONFIG["vector-collection-name"],
        scroll_filter=models.Filter(
            # should=[],
            must=[
                models.FieldCondition(
                    key='metadata."Food Product"', match=models.MatchText(text=query)),
            ],
            must_not=[
                models.FieldCondition(key="metadata.Ingredients", match=models.MatchText(text=allergen)) for allergen in allergens
            ],
        ),
        with_payload=['metadata."Food Product"',
                      "metadata.Ingredients", "metadata.Allergens"],
    )
    res = []
    for i in out[0]:
        ele = i.payload["metadata"]
        if ele not in res:
            res.append(ele)
    return res


if __name__ == "__main__":
    print(search_non_allergic_products("cookie", ["chocolate", "almond"]))
