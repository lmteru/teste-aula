# 📦 Importações necessárias
from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.datasets import load_iris
import mlflow


# 🔹 Inicializa a aplicação Flask
app = Flask(__name__)

# 🔹 Carrega o modelo salvo
with open("modelo_iris.pkl", "rb") as arquivo:
    modelo = pickle.load(arquivo)

# 🔹 Carrega nomes das classes (para traduzir a previsão)
iris = load_iris()
classes = iris.target_names

def prever_especie(valores):
    """
    Recebe um array ou lista de valores no formato:
    [comprimento_sépala, largura_sépala, comprimento_pétala, largura_pétala]
    e retorna o nome da espécie prevista.
    """
    with open("modelo_iris.pkl", "rb") as arquivo:
        modelo = pickle.load(arquivo)
    
    with open("name_labels.pkl", "rb") as arquivo:
        target_names = pickle.load(arquivo)

    valores = np.array(valores).reshape(1, -1)
    pred = modelo.predict(valores)
    especie = target_names[pred[0]]

    return especie

# 🔹 Define a rota de predição
@app.route("/predict", methods=["POST"])
def predict():
    """
    Espera um JSON com formato:
    {
        "data": [5.1, 3.5, 1.4, 0.2]
    }
    """
    try:
        # Lê os dados enviados via JSON
        dados = request.get_json()
        entradas = np.array(dados["data"])

        # Realiza a previsão
        predicoes = prever_especie(entradas)

        # Retorna o resultado em JSON
        return jsonify({
            "predicoes": predicoes
        })

    except Exception as e:
        return jsonify({
            "erro": str(e)
        }), 400

# 🔹 Executa a aplicação
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
