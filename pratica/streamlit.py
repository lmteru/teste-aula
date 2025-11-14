import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Distribuição das Classes - Iris")

# Upload ou leitura direta do arquivo
csv_path = st.text_input("Caminho do arquivo CSV:", "data.csv")

if csv_path:
    try:
        df = pd.read_csv(csv_path)

        st.subheader("Prévia do dataset:")
        st.write(df.head())

        # Contagem dos valores da coluna target
        counts = df["target"].value_counts()

        st.subheader("Distribuição da coluna target")

        # Gráfico de pizza
        fig, ax = plt.subplots()
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
        ax.axis("equal")  # deixa o gráfico circular

        st.pyplot(fig)

        # ------------------------------
        # 📊 Estatísticas das features
        # ------------------------------
        st.subheader("Estatísticas do Dataset")
        if 'target_name' in df.columns:
            df = df.drop('target_name', axis=1)

        # Seleciona apenas colunas numéricas
        stats_df = pd.DataFrame({
            "média": df.mean(),
            "desvio padrão": df.std(),
            "valor mínimo": df.min(),
            "valor máximo": df.max()
        })

        st.dataframe(stats_df)


    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
