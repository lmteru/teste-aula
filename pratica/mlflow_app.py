# 📦 Importações necessárias
from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.datasets import load_iris
import mlflow
import os
import pandas as pd
from sklearn import datasets

import logging

app = Flask(__name__)

def setup_logger(name, log_file, use_format=True, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    if use_format:
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def load_model(model_uri):
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        return loaded_model

    except Exception as e:
        app_logger.error(f"Erro ao baixar artefatos: {e}")
        return None

def prever_especie(entradas):
    """
    Recebe um array ou lista de valores no formato:
    [comprimento_sépala, largura_sépala, comprimento_pétala, largura_pétala]
    e retorna o nome da espécie prevista.
    """
    predictions = loaded_model.predict(entradas)

    return target_names[predictions]


@app.route("/invocations", methods=["POST"])
def predict():
    try:
        # Lê os dados enviados via JSON
        dados = request.get_json()
        entradas = np.array([dados["inputs"]])

        # Realiza a previsão
        predicoes = prever_especie(entradas)
        app_logger.info('SUCESSO')
        # Retorna o resultado em JSON
        return jsonify({
            "predictions": predicoes[0]
        })

    except Exception as e:
        app_logger.error(e)
        return jsonify({
            "erro": str(e)
        }), 400


if __name__ == "__main__":
    app_logger = setup_logger('main', 'app.log')
    model_uri = f"models:/iris_model/1"

    loaded_model = load_model(model_uri)
    iris = load_iris()
    target_names = iris.target_names

    app.run(host="0.0.0.0", port=5002, debug=True)
