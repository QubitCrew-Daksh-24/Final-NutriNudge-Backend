# Imports
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from accelerate import Accelerator
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json
import time

accelerator = Accelerator()

allergens_set = {'Soybeans', 'Eggs', 'Celery', 'Pine nuts', 'Peanuts', 'Almonds', 'Shellfish', 'Pork', 'Nuts', 'Anchovies',
                 'Mustard', 'Milk', 'Coconut', 'Strawberries', 'Alcohol', 'Chicken', 'Ghee', 'Fish', 'Cocoa', 'Wheat', 'Oats', 'Dairy', 'Rice'}

# Importing config file
CONFIG = None
with open(".\\..\\config.json") as f:
    CONFIG = json.load(f)
print("[-] Loaded configurations")

# Tokenizer Details
# model_name = CONFIG["token-model-name"]
model_kwargs = {"device": 'cuda'}
encode_kwargs = {"normalize_embeddings": False}

embeddings = HuggingFaceBgeEmbeddings(
    model_kwargs=model_kwargs,
    # model_name = model_name,
    encode_kwargs=encode_kwargs
)

print("[-] Embedding model initialised")

hist = ""

url = CONFIG["vector-db-url"]
api_key = CONFIG["vector-api-key"]
collection_name = CONFIG["vector-collection-name"]

client = QdrantClient(
    url=url,
    api_key=api_key,
    # prefer_grpc=True
)

print("[-] Qdrant Vector Database client started")

db = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name=collection_name
)

print("[-] Vector embeddings obtained")


def demo_vector_select():
    query = "List the products you have..."
    docs = db.similarity_search_with_score(query=query, k=5)
    doc, score = docs[0]
    print("##########################################################")
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})
    print("##########################################################")


cpt = """
<s>
[INST]
## System
You are an AI assistant for NutriNudge, an online retail shop specializing in food products. Your role is to offer a personalized shopping experience by suggesting suitable products for users, taking into account their specific allergies. You must filter out products that contain any allergens listed by the user.
The user specific allergens are given in "User Allergies" section. The products present are provided in the "Product Info" section.

## Product Info: 
{context}

------------------------------------------------------------------
## User Allergies:
$user_allergies$

Below provided is the list of you previous conversations, use it to give personilized reponses.
## Chat History:
$chat_history$
------------------------------------------------------------------
## Question: 
{question}

Only return the helpful answer below and nothing else.
[/INST]
## Helpful answer:
</s>
"""


def update_prompt(user_allergies):
    global cpt
    cpt = cpt.replace("$chat_history$", hist)
    print("[-] History added to prompt")
    cpt = cpt.replace("$user_allergies$", str(user_allergies))
    print("[-] Allergies intimated to model")
    prompt = PromptTemplate(template=cpt,
                            input_variables=['context', 'question'])
    print("[-] Prompt formatted")
    return prompt


def retrieval_qa_chain(llm, prompt, db, allergens):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(
                                               search_kwargs={'k': 10,
                                                              "filter": models.Filter(
                                                                  must_not=[
                                                                      models.FieldCondition(
                                                                          key="Ingredients",
                                                                          match=models.MatchAny(
                                                                              any=allergens,
                                                                          )
                                                                      )
                                                                  ]
                                                              ), }),
                                           return_source_documents=True,
                                           chain_type_kwargs={
                                               'prompt': prompt},
                                           )
    print("[-] QA chain initialised")

    return qa_chain


def load_llm():
    conf = {"max_new_tokens": 1024, "top_k": 5, "top_p": 0.80,
            "context_length": 5096, "gpu_layers": 10}
    llm = CTransformers(
        # model=CONFIG["model-link"],
        model="Intel/neural-chat-7b-v3-3",
        # model_type=CONFIG["model-name"],
        model_type="intel",
        temperature=0.3,
        config=conf
    )
    llm, conf = accelerator.prepare(llm, conf)
    # llm = AutoModelForCausalLM.from_pretrained(CONFIG["model-link"], model_type="mistral", gpu_layers=0, config=conf, local_files_only=True)
    print("[-] LLM loaded")
    return llm


def qa_bot(user_allergies):
    llm = load_llm()
    qa_prompt = update_prompt(user_allergies)
    qa = retrieval_qa_chain(llm, qa_prompt, db, user_allergies)
    print("[-] Chatbot is online")
    return qa


def final_result(query, user_allergies):
    global hist
    qa_result = qa_bot(user_allergies)
    print("[.] Generating recommendations for the query: "+query)
    res = qa_result.invoke({'query': query})
    print("[.] Response obtained")
    answer = res["result"]
    sources = res["source_documents"]
    with open("res.json", "w") as f:
        f.write(str(res))
    with open("answer.json", "w") as f:
        f.write(str(answer))
    with open("sources.json", "w") as f:
        print("sources:\n", sources)
        f.write(str(sources))
    hist += query + ":" + answer + "\n"
    rfcnc = "Look at the below suggested results..."
    components = []
    if sources:
        for source in sources:
            components.append({
                "title": source.metadata["Food Product"],
                "ingredients": source.metadata["Ingredients"]
            })
    else:
        rfcnc = ""
    response = answer+"\n"+rfcnc
    return {"response": response, "components": components}


def search_product(prompt, user_allergies):
    global cpt
    st = time.time()
    res = final_result(prompt, user_allergies)
    ed = time.time()
    print("&&&&"*10)
    print(cpt)
    print("===="*10)
    print(res)
    print("####"*10)
    print("Time Taken:", ed-st, "secs")
    print("!!!!"*10)
    print("[->] Responding...", res)
    print("[$$] Time Taken:", ed-st, "secs")
    return res


if __name__ == "__main__":
    # demo_vector_select()
    search_product("i want cookie", ["chocolate"])
    search_product("i want lemon", [])
    search_product("Hey do you guys offer refund?", ["chocolate"])
