from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from recipeBot import get_recipe
from search_better import search_non_allergic_products
from mistralBot import search_product

data = pd.read_csv("dataset.csv")

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return "This is backend"


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data["query"]
    allergens = data["allergens"]
    res = search_non_allergic_products(query, allergens)
    return jsonify(res), 200


@app.route("/recipe", methods=["POST"])
def recipe():
    data = request.get_json()
    ingredients = data["ingredients"]
    res = get_recipe(ingredients)
    return jsonify({"recipes": res}), 200


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data["query"]
    allergies = data["allergies"]
    res = search_product(query, allergies)
    return jsonify(res), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
