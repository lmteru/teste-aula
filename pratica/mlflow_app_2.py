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
        formatter = logging.Formatter(use_format)
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

def prever_especie(entradas, return_format=str):
    """
    Recebe um array ou lista de valores no formato:
    [comprimento_sépala, largura_sépala, comprimento_pétala, largura_pétala]
    e retorna o nome da espécie prevista.
    """
    predictions = loaded_model.predict(entradas)
    final = np.append(entradas, predictions[0])
    data_logger.info(','.join(['%.1f' % num for num in final]))
    if return_format is str:
        return target_names[predictions]
    else:
        return predictions[0]


@app.route("/invocations", methods=["POST"])
def predict_number():
    try:
        # Lê os dados enviados via JSON
        dados = request.get_json()
        app_logger.info('Chamado com: {}'.format(dados))
        entradas = np.array([dados["inputs"]])
        # Realiza a previsão
        predicoes = prever_especie(entradas)
        app_logger.info('SUCESSO NO PREDICT')
        # Retorna o resultado em JSON
        return jsonify({
            "predictions": predicoes[0]
        })

    except Exception as e:
        app_logger.error(e)
        return jsonify({
            "erro": str(e)
        }), 400


@app.route("/n_invocations", methods=["POST"])
def predict():
    try:
        # Lê os dados enviados via JSON
        dados = request.get_json()
        app_logger.info('Chamado com: {}'.format(dados))
        entradas = np.array([dados["inputs"]])

        # Realiza a previsão
        predicoes = prever_especie(entradas, int)
        print(predicoes)

        app_logger.info('SUCESSO NO PREDICT')
        # Retorna o resultado em JSON
        return jsonify({
            "predictions": str(predicoes)
        })

    except Exception as e:
        app_logger.error(e)
        return jsonify({
            "erro": str(e)
        }), 400

if __name__ == "__main__":
    app_logger = setup_logger('main', 'app.log', '%(asctime)s %(levelname)s %(message)s')
    data_logger = setup_logger('data', 'data.log', '%(message)s')
    model_uri = f"models:/iris_model/2"

    loaded_model = load_model(model_uri)
    iris = load_iris()
    target_names = iris.target_names

    app.run(host="0.0.0.0", port=5001, debug=True)
