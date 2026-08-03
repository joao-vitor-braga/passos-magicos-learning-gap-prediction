# Passos Mágicos — Sistema Preditivo de Risco de Defasagem

Projeto desenvolvido no contexto da **Pós-Graduação em Data Analytics (FIAP POSTECH — Datathon Fase 5)**, com foco em **análise exploratória, modelagem preditiva e deploy de aplicações de Machine Learning** aplicados à educação social.

O desafio consiste em identificar, a partir dos indicadores da **Pesquisa Extensiva do Desenvolvimento Educacional (PEDE)**, quais alunos da Associação Passos Mágicos apresentam maior **risco de entrar em defasagem escolar no ano seguinte**, permitindo priorizar acompanhamento pedagógico e psicossocial de forma proativa — antes que a defasagem efetivamente ocorra.

[Acessar deploy da aplicação - Passos Magicos - Learning Gap Prediction / Streamlit](https://pass0s-magicos-learning-gap-prediction.streamlit.app/)

---

## Como Executar o Projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/<joao-vitor-braga>/<passos-magicos-learning-gap-prediction>.git
cd <passos-magicos-learning-gap-prediction>
```

### 2. Criar e ativar um ambiente virtual (recomendado)

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Garantir a estrutura de pastas

Certifique-se de que os arquivos estão organizados conforme abaixo antes de executar:

```text
Datathon/
├── app/
│   ├── streamlit_app.py                    # Aplicação Streamlit principal
│   └── model/
│       ├── features_cat.joblib             # Lista de features categóricas
│       ├── features_num.joblib             # Lista de features numéricas
│       └── modelo_risco_defasagem.joblib   # Pipeline treinado (pré-processamento + modelo)
├── data/
│   ├── raw/
│   │   └── BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx   # Base bruta (3 abas: PEDE2022/2023/2024)
│   └── processed/
│       └── pede_painel_2022_2024.csv       # Painel longitudinal consolidado
├── notebooks/
│   ├── 01_data_cleaning.py                 # Limpeza e consolidação das 3 abas em um painel único
│   ├── 02_eda_analysis.ipynb               # Análise exploratória — responde às 11 perguntas do desafio
│   └── 03_predictive_model.ipynb           # Feature engineering, treino/teste, modelagem e avaliação
├── README.md
└── requirements.txt
```

### 5. Reproduzir a limpeza e a modelagem (opcional)

Se quiser reproduzir o pipeline de dados e o treinamento do zero:

```bash
cd notebooks
python 01_data_cleaning.py
```

Em seguida, abra e execute `02_eda_analysis.ipynb` e `03_predictive_model.ipynb` (Jupyter ou VS Code).

> Os artefatos gerados (`modelo_risco_defasagem.joblib`, `features_num.joblib`, `features_cat.joblib`) devem ser movidos para `app/model/`.

### 6. Executar a aplicação Streamlit

```bash
cd app
streamlit run app/streamlit_app.py
```

---

## Objetivo do Projeto

Construir um **sistema preditivo binário** capaz de estimar a probabilidade de um aluno apresentar **defasagem (moderada ou severa)** no ano seguinte, com base exclusivamente nos indicadores PEDE do **ano corrente**:

| Classe | Descrição |
|---|---|
| `0` — Sem risco | Aluno tende a permanecer em fase (`Defasagem > -1`) no ano seguinte |
| `1` — Em risco | Aluno tende a apresentar defasagem moderada ou severa (`Defasagem ≤ -1`) no ano seguinte |

O modelo deve permitir à equipe da Passos Mágicos identificar **sinais de alerta precoce**, possibilitando intervenção pedagógica e psicossocial antes que a defasagem se concretize.

---

## Dados Utilizados

- **Fonte**: Pesquisa Extensiva do Desenvolvimento Educacional (PEDE), Associação Passos Mágicos
- **Período**: 2022, 2023 e 2024 (3 abas com esquemas de colunas diferentes na base bruta)
- **Painel consolidado**: 3.030 registros (uma linha por aluno por ano), referentes a **1.661 alunos únicos**
- **Formato**: Excel bruto (`.xlsx`) → CSV consolidado após limpeza (`01_data_cleaning.py`)

### Principais indicadores do dataset PEDE

| Sigla | Significado |
|---|---|
| IAN | Adequação ao Nível |
| IDA | Desempenho Acadêmico |
| IEG | Engajamento |
| IAA | Autoavaliação |
| IPS | Psicossocial |
| IPP | Psicopedagógico (coletado só a partir de 2023) |
| IPV | Ponto de Virada |
| INDE | Índice de Desenvolvimento Educacional (nota geral ponderada) |
| Pedra | Classificação do aluno pelo INDE (Quartzo < Ágata < Ametista < Topázio) |

### Divisão dos Dados

Diferente de um split aleatório, adotamos um **split temporal**, que simula o uso real do modelo (prever o futuro com base no passado) e evita vazamento de informação entre anos:

- **Treino**: transição **2022 → 2023**
- **Teste**: transição **2023 → 2024** (690 observações, nunca vistas durante treino/tuning)
- **Total de observações utilizáveis** (alunos com ano seguinte consecutivo disponível): 1.365, com alvo praticamente balanceado (49,4% em risco / 50,6% sem risco)

---

## Metodologia

O projeto segue um pipeline completo de análise exploratória, engenharia de features e Machine Learning:

### 1. Análise Exploratória

O notebook `02_eda_analysis.ipynb` responde a 11 perguntas de negócio do desafio, entre elas:

- **Evolução do IAN**: o IAN médio sobe ano a ano (2022→2024) e a fatia de alunos severamente defasados cai de forma consistente.
- **IDA (desempenho acadêmico)**: não segue tendência monotônica como o IAN — oscila entre anos e fases.
- **IEG × IDA/IPV**: correlação positiva moderada (Pearson ≈ 0,54 e 0,56, respectivamente) — engajamento é fator relevante, mas não único.
- **IAA × desempenho real**: correlação fraca (0,12–0,13) — a autopercepção do aluno pouco acompanha seu desempenho e engajamento observados.
- **IPS como sinal de alerta**: diferença estatisticamente significativa para engajamento futuro, não significativa para desempenho acadêmico isoladamente (Mann-Whitney, p=0,139).
- **IPP × defasagem (IAN)**: diferença significativa entre categorias de defasagem (Kruskal-Wallis, p<0,0001), mas correlação fraca com IAN (Spearman = 0,13).
- **Drivers do IPV**: IDA e IEG lideram em 2022–2023; em 2024, IPP salta para a maior correlação com IPV (0,750).
- **Efetividade do programa (Pedra)**: Topázio quase dobra sua participação (15,1% → 24,9% → 30,9%) — o sinal mais forte de efetividade da análise.
- Diferenças por instituição de ensino, gênero e tempo de permanência na Passos Mágicos.

### 2. Feature Engineering

A variável-alvo (`Risco_Defasagem_Futuro`) é construída olhando o valor de `Defasagem` do **ano seguinte** para o mesmo aluno (chave `RA`), usando o painel longitudinal. As features são os indicadores do **ano corrente** — o momento em que a previsão seria feita na prática.

**12 features numéricas**: IAN, IDA, IEG, IAA, IPS, IPV, INDE, Fase_num, Idade, Anos_na_PM, Nota_Mat, Nota_Por
**2 features categóricas**: Genero, Instituicao_ensino

> **Nota:** o indicador **IPP** só passou a ser coletado a partir de 2023 e foi **excluído do conjunto de features**, para não descartar toda a safra de 2022 por valores nulos — mantendo o modelo treinável em ambas as transições disponíveis (2022→2023 e 2023→2024).

### 3. Pré-processamento

- Variáveis numéricas: `StandardScaler`
- Variáveis categóricas: `OneHotEncoder(handle_unknown="ignore")`
- Tudo encapsulado em um `Pipeline` do scikit-learn, garantindo que as mesmas transformações do treino sejam replicadas na inferência (usado diretamente pela aplicação Streamlit)

> **Limitação conhecida:** o `OneHotEncoder` de `Instituicao_ensino` foi ajustado apenas às categorias presentes no conjunto de treino (`Escola Pública`, `Rede Decisão`, `Escola JP II`). A base completa tem outras variações de preenchimento (`Pública`, `Privada`, `Privada - Programa de Apadrinhamento`, etc.) que, hoje, seriam tratadas como categoria desconhecida e ignoradas pelo modelo. Ver seção "Possíveis Melhorias".

### 4. Modelagem

Três algoritmos foram avaliados com **validação cruzada estratificada de 5 folds** sobre o conjunto de treino (transição 2022→2023):

| Modelo | AUC-ROC (CV 5-fold) |
|---|---|
| Regressão Logística (baseline) | 0,843 ± 0,041 |
| Random Forest | 0,822 ± 0,065 |
| **Gradient Boosting** | **0,854 ± 0,051** |

O **Gradient Boosting** apresentou o melhor AUC médio em validação cruzada e foi selecionado como modelo final, sendo então avaliado uma única vez sobre o conjunto de teste (transição 2023→2024).

---

## Resultados

Avaliado sobre o conjunto de teste (690 observações, transição 2023→2024, nunca vistas durante treino):

| Métrica | Valor |
|---|---|
| AUC-ROC | 0,825 |
| Acurácia | 0,75 |
| Precisão (classe "Em risco") | 0,75 |
| Recall (classe "Em risco") | 0,67 |
| F1-score (classe "Em risco") | 0,70 |

**Importância das variáveis** (top 5, Gradient Boosting): IAN (0,208), Fase_num (0,194), Idade (0,145), IPV (0,135), INDE (0,073) — os indicadores de trajetória e fase pesam mais na decisão do modelo do que os indicadores de desempenho pontual.

O recall de 67% para a classe "em risco" indica que o modelo ainda deixa de capturar cerca de 1 em cada 3 alunos que de fato entrarão em defasagem — o erro mais custoso no contexto social do programa. Ajuste de limiar de decisão é uma melhoria natural a ser explorada (ver seção de melhorias).

---

## Sobre a Aplicação Streamlit

A aplicação está organizada em quatro abas:

### Sobre o projeto
Contexto do desafio, glossário dos indicadores agrupado por tema (desempenho, trajetória, engajamento/bem-estar) e explicação de como interpretar o nível de risco (baixo/moderado/alto).

### Painel exploratório
Recorte navegável dos principais achados do EDA, com gráficos Altair (eixo de ano categórico e rótulos de valor sobre as barras):
- Evolução do IAN médio e da distribuição de categorias de defasagem por ano
- Efetividade por fase (Pedra) — evolução percentual de Quartzo/Ágata/Ametista/Topázio
- Relação entre engajamento (IEG) e desempenho (IDA)/ponto de virada (IPV)

### Predição de risco
- **Avaliação manual**: formulário com todos os indicadores do aluno, retornando probabilidade de risco, nível (🟢🟠🔴) e recomendação de acompanhamento.
- **Avaliação em lote**: upload de planilha CSV com múltiplos alunos, contagem por faixa de risco e download do resultado.

### Sobre o modelo
- Métricas recalculadas ao vivo sobre o conjunto de teste (AUC-ROC, acurácia, precisão, recall)
- Curva ROC
- Matriz de confusão (absoluta e normalizada por linha)
- Relatório de classificação completo (precisão/recall/F1 por classe)
- Importância das variáveis

---

## Métricas de Avaliação

- **AUC-ROC** — métrica principal de seleção do modelo, tanto em validação cruzada quanto no teste
- **Acurácia** — desempenho global
- **Precisão e Recall** (classe "em risco") — equilíbrio entre alertas desnecessários (falso positivo) e casos de risco não detectados (falso negativo, o mais custoso no contexto social do programa)
- **F1-score** — equilíbrio entre precisão e recall
- **Classification Report** — detalhamento por classe
- **Confusion Matrix** (absoluta e normalizada) — identificação dos padrões de erro do modelo

---

## Conclusões

- O **Gradient Boosting** superou Regressão Logística e Random Forest em validação cruzada, sendo selecionado como modelo final.
- **IAN, Fase_num, Idade e IPV** concentram a maior parte do peso decisório do modelo — indicadores de trajetória pesam mais do que notas pontuais.
- O **split temporal** (2022→2023 treino, 2023→2024 teste) foi decisivo para uma avaliação realista, evitando vazamento de informação entre anos e simulando o uso real do modelo.
- A exclusão do **IPP** (ausente em 2022) permitiu manter as duas transições disponíveis para treino e teste, ao custo de não usar um indicador que a análise exploratória mostrou ganhar relevância em 2024.
- O **recall de 67%** para a classe "em risco" é o principal ponto de atenção: quase 1/3 dos alunos que entrarão em defasagem não são sinalizados pelo modelo no limiar padrão de 50%.
- A restrição a **3 categorias de instituição de ensino** no encoder é uma limitação herdada do treinamento, não do aplicativo — ampliar o formulário sem retreinar o modelo não resolveria o problema.

---

## Possíveis Melhorias e Trabalhos Futuros

- Re-treinar o modelo com uma **taxonomia mais completa de instituição de ensino**, agrupando as ~12 variações de preenchimento da base (ex.: Pública / Privada / Programa de Apadrinhamento) em vez de restringir a 3 categorias.
- **Ajuste de limiar de decisão** (threshold tuning) para priorizar recall na classe "em risco", dado o custo social mais alto de um falso negativo.
- Reincorporar o **IPP** em um modelo específico para a transição 2023→2024, já que a análise exploratória indica sua relevância crescente.
- Reavaliação periódica (retreinamento) conforme novos ciclos do PEDE forem coletados, para monitorar deriva do modelo.
- Otimização de hiperparâmetros via `GridSearchCV`/Optuna e/ou ensemble dos três modelos testados.

---

## Tecnologias Utilizadas

- Python 3.11
- Streamlit
- Altair (visualizações interativas)
- scikit-learn
- pandas
- NumPy
- Matplotlib / Seaborn (notebooks de EDA e modelagem)
- statsmodels (testes estatísticos no EDA)
- joblib

---

## Observação

Este projeto foi desenvolvido **exclusivamente para fins acadêmicos**, como parte do Datathon PosTech (Fase 5 — Data Analytics), utilizando dados reais disponibilizados pela Associação Passos Mágicos para fins educacionais. A ferramenta é um apoio à priorização de acompanhamento pedagógico e psicossocial, não substituindo a avaliação da equipe da instituição.
