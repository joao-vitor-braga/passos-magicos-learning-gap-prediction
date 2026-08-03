"""
Aplicação Streamlit — Passos Mágicos
Previsão de Risco de Defasagem de Alunos

Deploy sugerido: Streamlit Community Cloud
Repositório: <preencher link do GitHub>

Estrutura esperada de pastas:
    app/
        app.py
        model/
            modelo_risco_defasagem.joblib
            features_num.joblib
            features_cat.joblib
    data/
        processed/
            pede_painel_2022_2024.csv
"""

import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(
    page_title="Passos Mágicos — Risco de Defasagem",
    page_icon="🧭",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Identidade visual
# ---------------------------------------------------------------------------
COR_AZUL = "#1B4F72"
COR_LARANJA = "#E8871E"
COR_VERDE = "#1E8449"
COR_VERMELHO = "#B03A2E"

PALETA_CATEGORICA = [COR_AZUL, COR_LARANJA, COR_VERDE, COR_VERMELHO]

st.markdown(
    f"""
    <style>
    .main-title {{
        color: {COR_AZUL};
        font-size: 2rem;
        font-weight: 800;
    }}
    .subtitle {{
        color: {COR_LARANJA};
        font-weight: 600;
    }}
    .stButton>button {{
        background-color: {COR_AZUL};
        color: white;
        font-weight: 700;
        border-radius: 8px;
    }}
    div[data-testid="stMetricValue"] {{
        color: {COR_AZUL};
    }}
    .card-titulo {{
        font-weight: 700;
        color: {COR_AZUL};
        font-size: 1.05rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "model")

_CANDIDATOS_DATA_PATH = [
    os.path.join(PROJECT_DIR, "data", "processed", "pede_painel_2022_2024.csv"),
    os.path.join(BASE_DIR, "data", "processed", "pede_painel_2022_2024.csv"),
    os.path.join(BASE_DIR, "data", "pede_painel_2022_2024.csv"),
]
DATA_PATH = next((p for p in _CANDIDATOS_DATA_PATH if os.path.exists(p)), _CANDIDATOS_DATA_PATH[0])

FEATURES_NUM = ["IAN", "IDA", "IEG", "IAA", "IPS", "IPV", "INDE",
                 "Fase_num", "Idade", "Anos_na_PM", "Nota_Mat", "Nota_Por"]
FEATURES_CAT = ["Genero", "Instituicao_ensino"]

# Categorias que o modelo efetivamente reconhece (definidas no treinamento em
# 03_predictive_model.ipynb). Categorias fora desta lista são tratadas pelo
# OneHotEncoder(handle_unknown="ignore") como "desconhecidas" e IGNORADAS
# silenciosamente — por isso o formulário só oferece estas opções.
INSTITUICOES_MODELO = ["Escola Pública", "Rede Decisão", "Escola JP II"]

GLOSSARIO_GRUPOS = {
    "📚 Desempenho acadêmico": {
        "IDA": "Desempenho acadêmico — média das notas escolares (Matemática, Português, Inglês).",
        "INDE": "Índice do Desenvolvimento Educacional — nota geral que resume os demais indicadores.",
    },
    "🎯 Trajetória e fase": {
        "IAN": "Adequação de nível — o quanto a fase atual do aluno está alinhada à idade/série esperada.",
        "IPV": "Ponto de virada — proximidade do aluno de atingir uma mudança de trajetória.",
        "Pedra": "Classificação de fase do programa (Quartzo → Ágata → Ametista → Topázio).",
        "Defasagem": "Diferença entre a fase atual e a fase ideal do aluno; valores negativos indicam atraso.",
    },
    "❤️ Engajamento e bem-estar": {
        "IEG": "Engajamento — participação e envolvimento do aluno nas atividades da Passos Mágicos.",
        "IAA": "Autoavaliação — percepção do próprio aluno sobre seu desempenho e evolução.",
        "IPS": "Psicossocial — indicadores de bem-estar emocional e social do aluno.",
    },
}

# ---------------------------------------------------------------------------
# Carregamento de modelo e dados (cacheados)
# ---------------------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    caminho_modelo = os.path.join(MODEL_DIR, "modelo_risco_defasagem.joblib")
    caminho_num = os.path.join(MODEL_DIR, "features_num.joblib")
    caminho_cat = os.path.join(MODEL_DIR, "features_cat.joblib")

    faltando = [p for p in [caminho_modelo, caminho_num, caminho_cat] if not os.path.exists(p)]
    if faltando:
        st.error(
            "Não encontrei os arquivos do modelo. Verifique se a pasta `model/` está "
            f"ao lado deste script (`{BASE_DIR}`) e contém os arquivos:\n\n"
            "- modelo_risco_defasagem.joblib\n- features_num.joblib\n- features_cat.joblib\n\n"
            f"Arquivo(s) não encontrado(s): {', '.join(os.path.basename(p) for p in faltando)}"
        )
        st.stop()

    modelo = joblib.load(caminho_modelo)
    features_num = joblib.load(caminho_num)
    features_cat = joblib.load(caminho_cat)
    return modelo, features_num, features_cat


@st.cache_data
def _ler_csv_painel(caminho: str):
    return pd.read_csv(caminho)


def carregar_painel():
    """
    A checagem de existência roda a cada execução (fora do cache), para não
    travar em um resultado 'não encontrado' de uma execução anterior, antes
    do CSV existir no caminho certo. Só a leitura em si é cacheada.
    """
    if not os.path.exists(DATA_PATH):
        return None
    return _ler_csv_painel(DATA_PATH)


@st.cache_data
def montar_conjunto_teste(painel: pd.DataFrame):
    """
    Reproduz a lógica de feature engineering e split temporal do notebook
    03_predictive_model.ipynb: usa a transição 2023->2024 (conjunto de teste
    definido no treinamento) para avaliar o modelo já treinado.
    """
    df = painel.sort_values(["RA", "Ano"]).reset_index(drop=True).copy()
    df["Defasagem_next"] = df.groupby("RA")["Defasagem"].shift(-1)
    df["Ano_next"] = df.groupby("RA")["Ano"].shift(-1)

    base = df[df["Ano_next"] - df["Ano"] == 1].copy()
    base["Risco_Defasagem_Futuro"] = (base["Defasagem_next"] <= -1).astype(int)

    modelo_df = base.dropna(subset=FEATURES_NUM + ["Risco_Defasagem_Futuro"]).copy()
    teste = modelo_df[modelo_df["Ano"] == 2023].copy()

    X_test = teste[FEATURES_NUM + FEATURES_CAT]
    y_test = teste["Risco_Defasagem_Futuro"]
    return X_test, y_test


modelo, _, _ = carregar_modelo()
painel = carregar_painel()

MENSAGEM_DADOS_FALTANDO = (
    "Não encontrei `pede_painel_2022_2024.csv`. Procurei em:\n\n"
    + "\n".join(f"- `{p}`" for p in _CANDIDATOS_DATA_PATH)
    + "\n\nColoque o arquivo em uma dessas pastas ao lado deste script para habilitar esta seção."
)

# ---------------------------------------------------------------------------
# Helpers de gráfico (Altair — eixo de ano categórico + rótulos de valor)
# ---------------------------------------------------------------------------
def grafico_linha_com_rotulos(serie: pd.Series, titulo_y: str, cor=COR_AZUL):
    df = serie.reset_index()
    df.columns = ["Ano", titulo_y]
    df["Ano"] = df["Ano"].astype(str)

    base = alt.Chart(df).encode(
        x=alt.X("Ano:N", title="Ano", axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{titulo_y}:Q", title=titulo_y),
    )
    linha = base.mark_line(color=cor, strokeWidth=3)
    pontos = base.mark_point(color=cor, size=80, filled=True)
    rotulos = base.mark_text(dy=-14, color=cor, fontWeight="bold").encode(
        text=alt.Text(f"{titulo_y}:Q", format=".2f")
    )
    return (linha + pontos + rotulos).properties(height=320)


def grafico_barras_agrupadas(df_wide: pd.DataFrame, titulo_y: str, formato="{:.1f}%"):
    """df_wide: índice = Ano, colunas = categorias, valores = %."""
    df_long = df_wide.reset_index().melt(id_vars="Ano", var_name="Categoria", value_name=titulo_y)
    df_long["Ano"] = df_long["Ano"].astype(str)
    df_long["rotulo"] = df_long[titulo_y].map(lambda v: formato.format(v))

    base = alt.Chart(df_long).encode(
        x=alt.X("Ano:N", title="Ano", axis=alt.Axis(labelAngle=0)),
        xOffset="Categoria:N",
        color=alt.Color("Categoria:N", scale=alt.Scale(range=PALETA_CATEGORICA), legend=alt.Legend(title="")),
    )
    barras = base.mark_bar().encode(y=alt.Y(f"{titulo_y}:Q", title=titulo_y))
    rotulos = base.mark_text(dy=-6, fontSize=11, fontWeight="bold").encode(
        y=alt.Y(f"{titulo_y}:Q"),
        text="rotulo:N",
        color=alt.value("#333333"),
    )
    return (barras + rotulos).properties(height=340)


def grafico_barras_simples(serie: pd.Series, titulo_x: str, cor=COR_AZUL, formato="{:.3f}"):
    df = serie.reset_index()
    df.columns = ["Variável", titulo_x]
    df["rotulo"] = df[titulo_x].map(lambda v: formato.format(v))

    base = alt.Chart(df).encode(
        y=alt.Y("Variável:N", sort="-x", title=""),
        x=alt.X(f"{titulo_x}:Q", title=titulo_x),
    )
    barras = base.mark_bar(color=cor)
    rotulos = base.mark_text(align="left", dx=4, fontWeight="bold", color="#333333").encode(text="rotulo:N")
    return (barras + rotulos).properties(height=32 * len(df) + 40)


def matriz_confusao_heatmap(cm: np.ndarray, labels: list, titulo: str, normalizada=False):
    linhas = []
    total_linha = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, total_linha, out=np.zeros_like(cm, dtype=float), where=total_linha != 0) * 100
    for i, real in enumerate(labels):
        for j, previsto in enumerate(labels):
            valor = cm_pct[i, j] if normalizada else cm[i, j]
            texto = f"{valor:.1f}%" if normalizada else f"{int(valor)}"
            linhas.append({"Real": real, "Previsto": previsto, "valor": valor, "texto": texto})
    df = pd.DataFrame(linhas)

    base = alt.Chart(df).encode(
        x=alt.X("Previsto:N", title="Previsto pelo modelo", sort=labels),
        y=alt.Y("Real:N", title="Valor real observado", sort=labels),
    )
    quadros = base.mark_rect().encode(
        color=alt.Color("valor:Q", scale=alt.Scale(scheme="blues"), legend=None)
    )
    limiar_cor = float(df["valor"].max() * 0.6)
    texto = base.mark_text(fontSize=16, fontWeight="bold").encode(
        text="texto:N",
        color=alt.condition(alt.datum.valor > limiar_cor, alt.value("white"), alt.value("black")),
    )
    return (quadros + texto).properties(height=260, title=titulo)


def cabecalho(titulo, subtitulo):
    st.markdown(f'<p class="main-title">{titulo}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">{subtitulo}</p>', unsafe_allow_html=True)
    st.divider()


# ---------------------------------------------------------------------------
# Cabeçalho fixo + navegação por abas
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">🧭 Passos Mágicos — Risco de Defasagem</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Datathon PosTech — Fase 5 | Base PEDE 2022–2024</p>',
    unsafe_allow_html=True,
)
st.divider()

aba_sobre, aba_painel, aba_predicao, aba_modelo = st.tabs(
    ["Sobre o projeto", "Painel exploratório", "Predição de risco", "Sobre o modelo"]
)

# ===========================================================================
# ABA 1 — Sobre o projeto
# ===========================================================================
with aba_sobre:
    st.write(
        "Este aplicativo foi desenvolvido para o **Datathon PosTech — Fase 5**, com base na "
        "série histórica **PEDE (2022–2024)** da Associação Passos Mágicos. Ele consolida em "
        "um só lugar os principais achados da análise exploratória e o modelo preditivo de "
        "risco de defasagem treinado sobre essa base."
    )

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown('<span class="card-titulo">🗂️ O que você encontra aqui</span>', unsafe_allow_html=True)
            st.markdown(
                "- **📊 Painel exploratório** — tendências dos indicadores ao longo dos anos.\n"
                "- **🎯 Predição de risco** — estimativa individual ou em lote (CSV).\n"
                "- **🧪 Sobre o modelo** — métricas de desempenho, calculadas sobre o "
                "conjunto de teste (transição 2023→2024)."
            )
    with col2:
        with st.container(border=True):
            st.markdown('<span class="card-titulo">🧭 Como interpretar o risco</span>', unsafe_allow_html=True)
            st.markdown(
                "O modelo estima a **probabilidade de um aluno apresentar defasagem "
                "(moderada ou severa) no ano seguinte**, a partir dos indicadores do ano "
                "corrente."
            )
            st.markdown(
                "🟢 **Baixo** (< 33%) &nbsp;·&nbsp; 🟠 **Moderado** (33–66%) &nbsp;·&nbsp; 🔴 **Alto** (> 66%)"
            )

    st.divider()
    st.markdown("#### 📖 Glossário dos indicadores")
    cols_glossario = st.columns(len(GLOSSARIO_GRUPOS))
    for col, (grupo, itens) in zip(cols_glossario, GLOSSARIO_GRUPOS.items()):
        with col:
            with st.container(border=True):
                st.markdown(f'<span class="card-titulo">{grupo}</span>', unsafe_allow_html=True)
                for termo, descricao in itens.items():
                    st.markdown(f"**{termo}**")
                    st.caption(descricao)

    st.info(
        "💡 Use as abas acima para navegar. Nenhum dado é enviado para fora desta sessão — "
        "os cálculos rodam localmente na aplicação."
    )

# ===========================================================================
# ABA 2 — Painel exploratório (EDA)
# ===========================================================================
with aba_painel:
    st.markdown("### 📊 Principais tendências identificadas na análise (PEDE 2022–2024)")
    st.caption("Escolha um dos recortes abaixo para explorar os achados do EDA.")

    if painel is None:
        st.error(MENSAGEM_DADOS_FALTANDO)
    else:
        secao = st.radio(
            "Seção do painel",
            [
                "📈 Evolução da adequação de nível",
                "💎 Efetividade por fase (Pedra)",
                "🤝 Engajamento x desempenho",
            ],
            horizontal=True,
            label_visibility="collapsed",
        )
        st.divider()

        if secao == "📈 Evolução da adequação de nível":
            st.markdown("**Pergunta:** o IAN médio e a proporção de alunos defasados vêm melhorando ao longo dos anos?")

            ian_ano = painel.groupby("Ano")["IAN"].mean().round(2)
            st.altair_chart(grafico_linha_com_rotulos(ian_ano, "IAN"), use_container_width=True)

            defasagem_contagem = (
                painel.dropna(subset=["Categoria_Defasagem"])
                .groupby(["Ano", "Categoria_Defasagem"])
                .size()
                .unstack(fill_value=0)
            )
            defasagem_pct = (defasagem_contagem.div(defasagem_contagem.sum(axis=1), axis=0) * 100).round(1)
            st.caption("Distribuição percentual de alunos por categoria de defasagem, por ano")
            st.altair_chart(grafico_barras_agrupadas(defasagem_pct, "Percentual"), use_container_width=True)

            st.success(
                "📌 **Leitura:** o IAN médio vem subindo ano a ano, e a fatia de alunos "
                "severamente defasados vem caindo — mesma direção indicada pelos gráficos acima."
            )

        elif secao == "💎 Efetividade por fase (Pedra)":
            st.markdown("**Pergunta:** a proporção de alunos nas fases mais avançadas (Ametista, Topázio) cresce ao longo do ciclo?")

            pedra_contagem = (
                painel.dropna(subset=["Pedra"])
                .groupby(["Ano", "Pedra"])
                .size()
                .unstack(fill_value=0)
            )
            pedra_pct = (pedra_contagem.div(pedra_contagem.sum(axis=1), axis=0) * 100).round(1)
            ordem_pedra = [p for p in ["Quartzo", "Ágata", "Ametista", "Topázio"] if p in pedra_pct.columns]
            st.altair_chart(grafico_barras_agrupadas(pedra_pct[ordem_pedra], "Percentual"), use_container_width=True)

            if "Topázio" in pedra_pct.columns:
                st.success(
                    "📌 **Leitura:** participação em Topázio por ano — "
                    + " → ".join(f"{ano}: {pedra_pct.loc[ano, 'Topázio']:.1f}%" for ano in pedra_pct.index)
                )

        else:
            st.markdown("**Pergunta:** alunos mais engajados (IEG) tendem a ter melhor desempenho (IDA) e maior IPV?")
            colc1, colc2 = st.columns(2)
            corr_ida = painel[["IEG", "IDA"]].corr().iloc[0, 1]
            corr_ipv = painel[["IEG", "IPV"]].corr().iloc[0, 1]
            colc1.metric("Correlação IEG × IDA", f"{corr_ida:.2f}")
            colc2.metric("Correlação IEG × IPV", f"{corr_ipv:.2f}")

            faixas = pd.cut(painel["IEG"], bins=[0, 4, 6, 8, 10], labels=["0–4", "4–6", "6–8", "8–10"])
            medias_faixa = painel.groupby(faixas, observed=True)[["IDA", "IPV"]].mean().round(2)
            medias_faixa.index.name = "Ano"  # reaproveita o helper de gráfico (rótulo genérico "Ano" -> faixa)
            st.caption("IDA e IPV médios por faixa de engajamento (IEG)")
            grafico = grafico_barras_agrupadas(medias_faixa, "Média").properties()
            grafico = grafico.encode(x=alt.X("Ano:N", title="Faixa de IEG", axis=alt.Axis(labelAngle=0)))
            st.altair_chart(grafico, use_container_width=True)

            st.success(
                "📌 **Leitura:** correlação positiva moderada — alunos mais engajados tendem a ter "
                "desempenho e ponto de virada mais altos, mas a relação não é determinística."
            )

# ===========================================================================
# ABA 3 — Predição de risco
# ===========================================================================
with aba_predicao:
    st.markdown("### 🎯 Estime a probabilidade de um aluno apresentar defasagem no próximo ano")

    st.write(
        "Preencha os indicadores do aluno referentes ao **ano corrente** para estimar a "
        "probabilidade de ele apresentar **defasagem (moderada ou severa) no ano seguinte**, "
        "com base no modelo treinado sobre a série histórica PEDE (2022–2024)."
    )

    modo = st.radio(
        "Como deseja utilizar a ferramenta?",
        ["Avaliar um aluno manualmente", "Avaliar uma planilha de alunos (CSV)"],
        horizontal=True,
    )

    def montar_dataframe_entrada(dados: dict) -> pd.DataFrame:
        return pd.DataFrame([dados])

    def exibe_resultado(proba: float):
        risco_pct = proba * 100
        if risco_pct >= 66:
            nivel, cor = "Alto", "🔴"
        elif risco_pct >= 33:
            nivel, cor = "Moderado", "🟠"
        else:
            nivel, cor = "Baixo", "🟢"

        st.metric("Probabilidade de risco de defasagem no próximo ano", f"{risco_pct:.1f}%")
        st.markdown(f"### {cor} Nível de risco: **{nivel}**")
        st.progress(min(max(proba, 0.0), 1.0))

        if nivel == "Alto":
            st.warning(
                "Recomenda-se priorizar este aluno para acompanhamento pedagógico e "
                "psicossocial reforçado no próximo ciclo."
            )
        elif nivel == "Moderado":
            st.info("Recomenda-se monitoramento próximo dos indicadores ao longo do próximo ciclo.")
        else:
            st.success("Aluno com indicadores consistentes com baixo risco de defasagem futura.")

    if modo == "Avaliar um aluno manualmente":
        with st.form("form_aluno"):
            st.subheader("Indicadores do aluno (ano corrente)")

            col1, col2 = st.columns(2)
            with col1:
                ian = st.slider("IAN — Adequação ao Nível", 0.0, 10.0, 7.0, 0.1)
                ida = st.slider("IDA — Desempenho Acadêmico", 0.0, 10.0, 6.5, 0.1)
                ieg = st.slider("IEG — Engajamento", 0.0, 10.0, 7.0, 0.1)
                iaa = st.slider("IAA — Autoavaliação", 0.0, 10.0, 7.0, 0.1)
                ips = st.slider("IPS — Psicossocial", 0.0, 10.0, 6.5, 0.1)
                ipv = st.slider("IPV — Ponto de Virada", 0.0, 10.0, 6.5, 0.1)
            with col2:
                inde = st.slider("INDE — Nota Geral", 0.0, 10.0, 6.8, 0.1)
                fase_num = st.number_input("Fase atual (número, 0 = Alfa)", min_value=0, max_value=9, value=3)
                idade = st.number_input("Idade", min_value=5, max_value=25, value=12)
                anos_pm = st.number_input("Anos na Passos Mágicos", min_value=0, max_value=15, value=2)
                nota_mat = st.slider("Nota de Matemática", 0.0, 10.0, 6.5, 0.1)
                nota_por = st.slider("Nota de Português", 0.0, 10.0, 6.5, 0.1)

            col3, col4 = st.columns(2)
            with col3:
                genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
            with col4:
                instituicao = st.selectbox("Instituição de ensino", INSTITUICOES_MODELO)
                st.caption(
                    "⚠️ Apenas estas 3 categorias foram reconhecidas no treinamento do modelo; "
                    "outras instituições da base seriam ignoradas pelo modelo (ver aba 'Sobre o modelo')."
                )

            submitted = st.form_submit_button("Calcular risco")

        if submitted:
            dados = {
                "IAN": ian, "IDA": ida, "IEG": ieg, "IAA": iaa, "IPS": ips, "IPV": ipv,
                "INDE": inde, "Fase_num": fase_num, "Idade": idade, "Anos_na_PM": anos_pm,
                "Nota_Mat": nota_mat, "Nota_Por": nota_por,
                "Genero": genero, "Instituicao_ensino": instituicao,
            }
            X_novo = montar_dataframe_entrada(dados)[FEATURES_NUM + FEATURES_CAT]
            proba = modelo.predict_proba(X_novo)[0, 1]
            st.divider()
            exibe_resultado(proba)

    else:
        st.subheader("Upload de planilha (CSV)")
        st.caption(
            "O arquivo deve conter as colunas: " + ", ".join(FEATURES_NUM + FEATURES_CAT)
        )
        st.caption(
            "⚠️ Para `Instituicao_ensino`, apenas os valores "
            + ", ".join(f"'{i}'" for i in INSTITUICOES_MODELO)
            + " são reconhecidos pelo modelo; outros valores serão tratados como categoria "
            "desconhecida (sem efeito na predição)."
        )
        with st.expander("Ver exemplo de formato esperado"):
            exemplo = pd.DataFrame([{
                "IAN": 7.0, "IDA": 6.5, "IEG": 7.0, "IAA": 7.0, "IPS": 6.5, "IPV": 6.5,
                "INDE": 6.8, "Fase_num": 3, "Idade": 12, "Anos_na_PM": 2,
                "Nota_Mat": 6.5, "Nota_Por": 6.5, "Genero": "Feminino",
                "Instituicao_ensino": "Escola Pública",
            }])
            st.dataframe(exemplo, hide_index=True, use_container_width=True)

        arquivo = st.file_uploader("Selecione o arquivo CSV", type=["csv"])

        if arquivo is not None:
            try:
                df_novo = pd.read_csv(arquivo)
                faltantes = [c for c in FEATURES_NUM + FEATURES_CAT if c not in df_novo.columns]
                if faltantes:
                    st.error(f"Colunas faltando no arquivo: {faltantes}")
                else:
                    probas = modelo.predict_proba(df_novo[FEATURES_NUM + FEATURES_CAT])[:, 1]
                    df_novo["Probabilidade_Risco_Defasagem"] = (probas * 100).round(1)
                    df_novo["Nivel_Risco"] = pd.cut(
                        probas,
                        bins=[-0.01, 0.33, 0.66, 1.01],
                        labels=["Baixo", "Moderado", "Alto"],
                    )
                    st.success(f"{len(df_novo)} alunos avaliados com sucesso.")

                    colr1, colr2, colr3 = st.columns(3)
                    contagem = df_novo["Nivel_Risco"].value_counts()
                    colr1.metric("🟢 Risco baixo", int(contagem.get("Baixo", 0)))
                    colr2.metric("🟠 Risco moderado", int(contagem.get("Moderado", 0)))
                    colr3.metric("🔴 Risco alto", int(contagem.get("Alto", 0)))

                    st.dataframe(
                        df_novo[["Probabilidade_Risco_Defasagem", "Nivel_Risco"] +
                                [c for c in df_novo.columns if c not in
                                 ["Probabilidade_Risco_Defasagem", "Nivel_Risco"]]],
                        use_container_width=True,
                    )
                    st.download_button(
                        "Baixar resultado em CSV",
                        df_novo.to_csv(index=False).encode("utf-8-sig"),
                        file_name="alunos_risco_defasagem.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Erro ao processar o arquivo: {e}")

# ===========================================================================
# ABA 4 — Sobre o modelo
# ===========================================================================
with aba_modelo:
    st.markdown("### 🧪 Como o modelo foi treinado e avaliado")

    st.markdown(
        "**Algoritmo:** Gradient Boosting Classifier, dentro de um `Pipeline` com padronização "
        "de variáveis numéricas e one-hot encoding de categóricas.\n\n"
        "**Alvo:** o aluno apresentar defasagem (`Defasagem ≤ -1`) no ano seguinte ao ano de referência.\n\n"
        "**Validação:** split temporal — treino na transição **2022→2023**, teste na transição "
        "**2023→2024** (dados nunca vistos durante o treino), simulando o uso real do modelo."
    )
    st.divider()

    if painel is None:
        st.warning(MENSAGEM_DADOS_FALTANDO)
    else:
        X_test, y_test = montar_conjunto_teste(painel)

        if len(X_test) == 0:
            st.warning("Não há observações suficientes na base para recalcular o conjunto de teste.")
        else:
            y_proba = modelo.predict_proba(X_test)[:, 1]
            y_pred = modelo.predict(X_test)

            auc = roc_auc_score(y_test, y_proba)
            relatorio = classification_report(
                y_test, y_pred, target_names=["Sem risco", "Em risco"], output_dict=True
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("AUC-ROC (teste)", f"{auc:.3f}")
            col2.metric("Acurácia", f"{relatorio['accuracy']:.2f}")
            col3.metric("Precisão (em risco)", f"{relatorio['Em risco']['precision']:.2f}")
            col4.metric("Recall (em risco)", f"{relatorio['Em risco']['recall']:.2f}")
            st.caption(f"Métricas recalculadas sobre {len(X_test)} observações da transição 2023→2024.")

            secao_modelo = st.radio(
                "Detalhamento",
                [
                    "Curva ROC",
                    "Matriz de confusão",
                    "Relatório de classificação",
                    "Importância das variáveis",
                ],
                horizontal=True,
                label_visibility="collapsed",
            )
            st.divider()

            if secao_modelo == "Curva ROC":
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
                linha_diag = pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})
                chart_roc = alt.Chart(roc_df).mark_line(color=COR_AZUL, strokeWidth=3).encode(
                    x=alt.X("fpr:Q", title="Taxa de falso positivo"),
                    y=alt.Y("tpr:Q", title="Taxa de verdadeiro positivo"),
                )
                chart_diag = alt.Chart(linha_diag).mark_line(
                    color="gray", strokeDash=[4, 4]
                ).encode(x="fpr:Q", y="tpr:Q")
                st.altair_chart((chart_roc + chart_diag).properties(height=360), use_container_width=True)
                st.caption(f"AUC-ROC = {auc:.3f} — quanto mais próxima de 1, melhor a separação entre alunos em risco e sem risco.")

            elif secao_modelo == "Matriz de confusão":
                cm = confusion_matrix(y_test, y_pred)
                labels = ["Sem risco", "Em risco"]

                colm1, colm2 = st.columns(2)
                with colm1:
                    st.altair_chart(
                        matriz_confusao_heatmap(cm, labels, "Valores absolutos", normalizada=False),
                        use_container_width=True,
                    )
                with colm2:
                    st.altair_chart(
                        matriz_confusao_heatmap(cm, labels, "Normalizada por linha (%)", normalizada=True),
                        use_container_width=True,
                    )
                st.caption(
                    "Linhas = valor real observado no ano seguinte; colunas = previsão do modelo. "
                    "A versão normalizada mostra, de cada grupo real, que percentual foi classificado em cada previsão."
                )

            elif secao_modelo == "Relatório de classificação":
                rel_df = pd.DataFrame(relatorio).T
                rel_df = rel_df.rename(index={"accuracy": "acurácia (global)"})
                st.dataframe(
                    rel_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}),
                    use_container_width=True,
                )
                st.caption(
                    "**Precisão**: dos previstos como 'em risco', quantos realmente estavam. "
                    "**Recall**: dos que realmente estavam em risco, quantos o modelo capturou. "
                    "**Support**: número de alunos em cada classe no conjunto de teste."
                )

            else:
                try:
                    clf = modelo.named_steps["clf"]
                    nomes_features = modelo.named_steps["prep"].get_feature_names_out()
                    importancias = (
                        pd.Series(clf.feature_importances_, index=nomes_features)
                        .sort_values(ascending=False)
                        .head(12)
                    )
                    st.altair_chart(
                        grafico_barras_simples(importancias, "Importância", cor=COR_LARANJA),
                        use_container_width=True,
                    )
                    st.caption("Variáveis com maior peso na decisão do modelo (Gradient Boosting).")
                except Exception:
                    st.info("Este modelo não expõe importância de variáveis diretamente.")

    st.divider()
    st.caption(
        "Desenvolvido para o Datathon PosTech — Fase 5, com a base PEDE da Associação Passos Mágicos."
    )
