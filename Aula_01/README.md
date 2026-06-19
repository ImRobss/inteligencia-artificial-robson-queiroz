# Aula 01 — IA Aplicada + Hello Modelo (Iris)

**Disciplina:** Inteligência Artificial  
**Professor:** Marcelo Batista

---

## Objetivos Principais

1. ✅ Entender o ambiente do Google Colab (CPU/GPU)
2. ✅ Manipular dados com **Pandas** utilizando o dataset Iris
3. ✅ Treinar o **primeiro modelo de ML** (baseline)
4. ✅ Entender o fluxo sagrado: **Dados → Treino → Teste**
5. ✅ Aprender sobre **pipelines reprodutíveis** (pré-processamento + modelo)
6. ✅ Resolver exercícios práticos

---

## Dataset Iris: "Hello World" da IA

### O que é?

- Dataset clássico com **150 amostras** de flores **Iris**
- **3 espécies**: Setosa, Versicolor, Virginica
- **4 features** (características) numéricas:
  - `sepal_length` (comprimento da sépala)
  - `sepal_width` (largura da sépala)
  - `petal_length` (comprimento da pétala)
  - `petal_width` (largura da pétala)
- **1 target** (alvo): espécie da flor

### Distribuição

- Cada espécie possui 50 amostras (dataset equilibrado)
- Classes são **linearmente separáveis** → ótimo para demonstração

---

## Conceitos-Chave

### X (Features) vs y (Target)

- **X**: As 4 medidas da flor (entrada do modelo)
- **y**: A espécie da flor (saída esperada - o que queremos prever)

### Treino vs Teste (Conceito Crítico)

| Fase       | Propósito                | Dados | Risco                                 |
| ---------- | ------------------------ | ----- | ------------------------------------- |
| **Treino** | Modelo "aprende" padrões | 80%   | Pode decorar os dados (overfitting)   |
| **Teste**  | Avalia generalização     | 20%   | Simula desempenho em dados não vistos |

**Por que são separados?** Se avalarmos no treino, podemos nos "enganar" (overfitting).

### DecisionTreeClassifier

- Algoritmo de **classificação** que cria uma árvore de decisão
- Aprende uma série de regras "se-então" a partir dos dados
- **Vantagens**: Intuitiva, interpretável, versátil
- **Base** para algoritmos mais avançados (Random Forests, Gradient Boosting)

---

## Pipeline: O Fluxo Sagrado

```
┌──────────────────────────────────────────────────────────────┐
│ 1️⃣  DADOS (Load Dataset)                                      │
│    └─ Iris dataset (150 amostras, 4 features, 1 target)      │
├──────────────────────────────────────────────────────────────┤
│ 2️⃣  EXPLORAÇÃO (EDA - Exploratory Data Analysis)             │
│    └─ shape, info(), isna().sum(), value_counts()            │
├──────────────────────────────────────────────────────────────┤
│ 3️⃣  PREPARAÇÃO (Separar X e y)                               │
│    └─ X = features | y = target                              │
├──────────────────────────────────────────────────────────────┤
│ 4️⃣  SPLIT (Dividir Treino/Teste)                             │
│    └─ test_size=0.2 (80% treino, 20% teste)                  │
│    └─ random_state=42 (reprodutibilidade)                    │
├──────────────────────────────────────────────────────────────┤
│ 5️⃣  TREINO (Fit Model)                                        │
│    └─ modelo.fit(X_train, y_train)                           │
├──────────────────────────────────────────────────────────────┤
│ 6️⃣  PREDIÇÃO (Evaluate)                                       │
│    └─ pred = modelo.predict(X_test)                          │
├──────────────────────────────────────────────────────────────┤
│ 7️⃣  MÉTRICAS (Measure Performance)                           │
│    └─ accuracy = accuracy_score(y_test, pred)                │
└──────────────────────────────────────────────────────────────┘
```

---

## Resultados Observados

### Comparação por `test_size` (random_state=42)

| test_size                   | Acurácia    |
| --------------------------- | ----------- |
| 0.2 (80% treino, 20% teste) | **100.00%** |
| 0.3 (70% treino, 30% teste) | **100.00%** |

**Insight:** Ambos atingiram acurácia perfeita, refletindo a **natureza separável do dataset Iris**.

### Comparação por `random_state` (test_size=0.2)

| random_state | Acurácia    |
| ------------ | ----------- |
| 42           | **100.00%** |
| 7            | **90.00%**  |

**Insight:** A escolha da **seed aleatória** impacta quais amostras caem no treino vs teste, afetando o resultado.

---

## Importante: 100% de Acurácia Nem Sempre é "Bom"

### Possíveis Causas:

1. **Split "sortudo"**: Sorte na divisão aleatória
2. **Overfitting**: O modelo "decorou" padrões do treino
3. **Vazamento de dados (leakage)**: Informação do teste vaza para o treino
4. **Dataset "fácil"**: Classes são claramete separáveis (comum no Iris)

### Iris é Um Caso Especial:

- Dataset é **relativamente simples** e bem-separado
- A espécie "Setosa" é quase sempre **perfeitamente distinguível** das outras
- DecisionTree é **muito adequada** para este tipo de problema
- 100% é mais um **reflexo da simplicidade dos dados** do que um problema

---

## Exercícios Práticos

### Exercício A — Entendendo os Dados

**Objetivo:** Exploração Exploratory Data Analysis (EDA)

**Código:**

```python
print("df.shape:", df.shape)  # Dimensões
print("df.info():")           # Tipos e info
df.info()
print("df.isna().sum():")     # Valores faltantes
print(df.isna().sum())
```

**Resultado:**

- `shape`: (150, 5) — 150 amostras, 5 colunas
- Sem valores faltantes (`isna().sum()` = 0)
- Distribuição: 50 amostras por espécie

**Aprendizado:**

- **X (features)**: Medidas da flor (sepal_length, sepal_width, petal_length, petal_width)
- **y (target)**: Espécie da flor (setosa, versicolor, virginica)

---

### Exercício B — Seu Primeiro Baseline

**Objetivo:** Entender o impacto do train/test split na acurácia

#### Parte 1: Impacto de `test_size`

```python
# test_size=0.2
X_train_02, X_test_02, y_train_02, y_test_02 = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Acurácia: 100.00%

# test_size=0.3
X_train_03, X_test_03, y_train_03, y_test_03 = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Acurácia: 100.00%
```

#### Parte 2: Impacto de `random_state`

```python
# random_state=42
# Acurácia: 100.00%

# random_state=7
# Acurácia: 90.00%
```

**Aprendizado:**

- A divisão treino/teste é **aleatória** → usar `random_state` para reprodutibilidade
- Diferentes seeds geram diferentes divisões
- Diferentes divisões podem resultar em acurácias diferentes

---

## Bibliotecas Usadas

| Biblioteca       | Função                                             |
| ---------------- | -------------------------------------------------- |
| **Pandas**       | Manipulação e análise de dados ("Excel do Python") |
| **Seaborn**      | Datasets prontos + visualizações estatísticas      |
| **Scikit-Learn** | ML clássico (split, modelos, métricas, pipelines)  |

---

## Recursos Adicionais

- [Dataset Iris no Kaggle](https://www.kaggle.com/datasets/uciml/iris)
- [Sklearn Decision Tree Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Pandas Documentation](https://pandas.pydata.org/)

---

**Última atualização:** Março 2026
