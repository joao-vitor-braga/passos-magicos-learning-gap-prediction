"""
01_data_cleaning.py
=====================
Datathon Passos Mágicos - Fase 5 PosTech

Objetivo: consolidar as 3 abas da base bruta (PEDE2022, PEDE2023, PEDE2024),
que possuem esquemas de colunas diferentes, em um único painel longitudinal
(uma linha por aluno por ano), padronizando nomes de colunas e categorias.

Saída: data/processed/pede_painel_2022_2024.csv
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx"
OUT_PATH = BASE_DIR / "data" / "processed" / "pede_painel_2022_2024.csv"

# ---------------------------------------------------------------------------
# 1. Carregar as 3 abas
# ---------------------------------------------------------------------------
xls = pd.ExcelFile(RAW_PATH)
d22 = pd.read_excel(xls, sheet_name="PEDE2022")
d23 = pd.read_excel(xls, sheet_name="PEDE2023")
d24 = pd.read_excel(xls, sheet_name="PEDE2024")

# ---------------------------------------------------------------------------
# 2. Padronização de valores categóricos
# ---------------------------------------------------------------------------
GENERO_MAP = {
    "Menina": "Feminino", "Menino": "Masculino",
    "Feminino": "Feminino", "Masculino": "Masculino",
}

def normaliza_genero(s):
    return s.map(GENERO_MAP).fillna(s)


def normaliza_pedra(s):
    """Corrige grafias inconsistentes e remove valores inválidos (ex.: 'INCLUIR')."""
    s = s.astype(str).str.strip()
    s = s.replace({"Agata": "Ágata", "nan": np.nan, "None": np.nan})
    s = s.where(s.isin(["Quartzo", "Ágata", "Ametista", "Topázio"]), np.nan)
    return s


def extrai_fase_numero(valor):
    """
    Extrai o número da fase a partir de formatos heterogêneos:
    - 2022: já é numérico (0-7)
    - 2023: texto 'ALFA', 'FASE 1' ... 'FASE 8'
    - 2024: texto 'ALFA', '1A', '2B', '8F', ou número solto (9)
    Retorna um inteiro (0 = ALFA) ou NaN se não for possível interpretar.
    """
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, (int, float)):
        return int(valor)
    valor = str(valor).strip().upper()
    if valor == "ALFA":
        return 0
    m = re.search(r"FASE\s*(\d+)", valor)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+)", valor)  # ex: '1A', '8F', '9'
    if m:
        return int(m.group(1))
    return np.nan


# ---------------------------------------------------------------------------
# 3. Renomear colunas de cada ano para um esquema comum
# ---------------------------------------------------------------------------
rename_22 = {
    "RA": "RA", "Fase": "Fase_raw", "Turma": "Turma", "Nome": "Nome",
    "Idade 22": "Idade", "Gênero": "Genero", "Ano ingresso": "Ano_ingresso",
    "Instituição de ensino": "Instituicao_ensino", "Pedra 22": "Pedra",
    "INDE 22": "INDE", "IAA": "IAA", "IEG": "IEG", "IPS": "IPS",
    "IDA": "IDA", "Matem": "Nota_Mat", "Portug": "Nota_Por", "Inglês": "Nota_Ing",
    "Indicado": "Indicado_bolsa", "Atingiu PV": "Atingiu_PV", "IPV": "IPV",
    "IAN": "IAN", "Fase ideal": "Fase_ideal_raw", "Defas": "Defasagem",
    "Rec Psicologia": "Rec_Psicologia",
}
rename_23 = {
    "RA": "RA", "Fase": "Fase_raw", "Turma": "Turma", "Nome Anonimizado": "Nome",
    "Idade": "Idade", "Gênero": "Genero", "Ano ingresso": "Ano_ingresso",
    "Instituição de ensino": "Instituicao_ensino", "Pedra 2023": "Pedra",
    "INDE 2023": "INDE", "IAA": "IAA", "IEG": "IEG", "IPS": "IPS", "IPP": "IPP",
    "IDA": "IDA", "Mat": "Nota_Mat", "Por": "Nota_Por", "Ing": "Nota_Ing",
    "Indicado": "Indicado_bolsa", "Atingiu PV": "Atingiu_PV", "IPV": "IPV",
    "IAN": "IAN", "Fase Ideal": "Fase_ideal_raw", "Defasagem": "Defasagem",
    "Rec Psicologia": "Rec_Psicologia",
}
rename_24 = {
    "RA": "RA", "Fase": "Fase_raw", "Turma": "Turma", "Nome Anonimizado": "Nome",
    "Idade": "Idade", "Gênero": "Genero", "Ano ingresso": "Ano_ingresso",
    "Instituição de ensino": "Instituicao_ensino", "Pedra 2024": "Pedra",
    "INDE 2024": "INDE", "IAA": "IAA", "IEG": "IEG", "IPS": "IPS", "IPP": "IPP",
    "IDA": "IDA", "Mat": "Nota_Mat", "Por": "Nota_Por", "Ing": "Nota_Ing",
    "Indicado": "Indicado_bolsa", "Atingiu PV": "Atingiu_PV", "IPV": "IPV",
    "IAN": "IAN", "Fase Ideal": "Fase_ideal_raw", "Defasagem": "Defasagem",
    "Rec Psicologia": "Rec_Psicologia", "Escola": "Escola_2024",
}

COMMON_COLS = [
    "RA", "Nome", "Idade", "Genero", "Ano_ingresso", "Instituicao_ensino",
    "Fase_raw", "Turma", "Pedra", "INDE", "IAA", "IEG", "IPS", "IPP", "IDA",
    "Nota_Mat", "Nota_Por", "Nota_Ing", "Indicado_bolsa", "Atingiu_PV",
    "IPV", "IAN", "Fase_ideal_raw", "Defasagem", "Rec_Psicologia",
]


def prepara_ano(df, rename_map, ano):
    d = df.rename(columns=rename_map).copy()
    for col in COMMON_COLS:
        if col not in d.columns:
            d[col] = np.nan
    d = d[COMMON_COLS].copy()
    d["Ano"] = ano
    d["Genero"] = normaliza_genero(d["Genero"])
    d["Pedra"] = normaliza_pedra(d["Pedra"])
    d["Fase_num"] = d["Fase_raw"].apply(extrai_fase_numero)
    d["Fase_ideal_num"] = d["Fase_ideal_raw"].apply(extrai_fase_numero)
    # Padroniza Indicado_bolsa e Atingiu_PV para booleano quando possível
    for col in ["Indicado_bolsa", "Atingiu_PV"]:
        d[col] = d[col].map({"Sim": True, "Não": False}).where(
            d[col].isin(["Sim", "Não"]), np.nan
        )
    return d


p22 = prepara_ano(d22, rename_22, 2022)
p23 = prepara_ano(d23, rename_23, 2023)
p24 = prepara_ano(d24, rename_24, 2024)

painel = pd.concat([p22, p23, p24], ignore_index=True)

# ---------------------------------------------------------------------------
# 4. Conversões numéricas e checagens de consistência
# ---------------------------------------------------------------------------
def corrige_idade(valor):
    """
    A coluna 'Idade' de 2023 veio com valores corrompidos pelo Excel:
    idades pequenas (ex.: 8) foram convertidas em datas (ex.: 1900-01-08).
    Nesses casos, o dia da data corresponde à idade original.
    """
    if isinstance(valor, pd.Timestamp):
        return valor.day
    try:
        import datetime
        if isinstance(valor, datetime.datetime):
            return valor.day
    except Exception:
        pass
    return valor

painel["Idade"] = painel["Idade"].apply(corrige_idade)

num_cols = ["INDE", "IAA", "IEG", "IPS", "IPP", "IDA", "Nota_Mat", "Nota_Por",
            "Nota_Ing", "IPV", "IAN", "Defasagem", "Idade", "Ano_ingresso"]
for c in num_cols:
    painel[c] = pd.to_numeric(painel[c], errors="coerce")

# Categoria de defasagem (para facilitar leitura gerencial)
def categoriza_defasagem(x):
    if pd.isna(x):
        return np.nan
    if x >= 0:
        return "Em fase"
    elif x >= -2:      # -2 <= x < 0
        return "Moderada"
    else:               # x < -2
        return "Severa"

painel["Categoria_Defasagem"] = painel["Defasagem"].apply(categoriza_defasagem)

# Tempo de casa (anos na Passos Mágicos, aproximado)
painel["Anos_na_PM"] = painel["Ano"] - painel["Ano_ingresso"]
painel.loc[painel["Anos_na_PM"] < 0, "Anos_na_PM"] = np.nan

# Reordena colunas
painel = painel[
    ["RA", "Nome", "Ano", "Idade", "Genero", "Ano_ingresso", "Anos_na_PM",
     "Instituicao_ensino", "Fase_raw", "Fase_num", "Turma", "Pedra", "INDE",
     "IAN", "IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "Nota_Mat", "Nota_Por",
     "Nota_Ing", "Fase_ideal_raw", "Fase_ideal_num", "Defasagem",
     "Categoria_Defasagem", "Indicado_bolsa", "Atingiu_PV", "Rec_Psicologia"]
]

painel = painel.sort_values(["RA", "Ano"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 5. Relatório rápido de qualidade dos dados
# ---------------------------------------------------------------------------
print("Shape final do painel:", painel.shape)
print("\nAlunos únicos:", painel["RA"].nunique())
print("\nRegistros por ano:\n", painel["Ano"].value_counts().sort_index())
print("\n% nulos por coluna:\n", (painel.isnull().mean() * 100).round(1))

painel.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print(f"\nArquivo salvo em: {OUT_PATH}")