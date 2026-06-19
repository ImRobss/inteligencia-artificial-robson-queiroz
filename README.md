# Inteligência Artificial — Repositório da Disciplina

Repositório criado para a disciplina de **Inteligência Artificial** da faculdade, ministrada pelo **Prof. Marcelo Batista**. Aqui estão organizados os materiais de pré-aula, exercícios práticos e o projeto final da disciplina.

---

## Objetivo

Ao longo da disciplina me propus a sair do zero em Machine Learning e construir, do início ao fim, uma aplicação real capaz de tomar decisões baseadas em dados. O caminho foi progressivo: entender os fundamentos teóricos, aplicar em datasets clássicos, e culminar num projeto completo com modelo preditivo integrado a uma API e interface web.

---

## Estrutura do Repositório

```
.
├── Pre_Aula_01/        # Preparação: fundamentos de Python para dados
├── Aula_01/            # Primeiro modelo de ML com dataset Iris (DecisionTree)
├── Pre_Aula_02/        # Revisão de Pandas, correlação e visualização
├── Aula_02/            # Pipeline com dataset Titanic
└── agente-saude/       # Projeto final — Agente de triagem para diabetes
```

---

## Aulas

### Pré-Aula 01 — Fundamentos de Python para Dados
Revisão de bibliotecas essenciais (NumPy, Pandas, Matplotlib/Seaborn) e manipulação de DataFrames antes de entrar no ciclo de ML.

### Aula 01 — Primeiro Modelo de ML (Dataset Iris)
Primeiro contato com o fluxo completo de Machine Learning:
- Exploração de dados (EDA)
- Separação treino/teste com `train_test_split`
- Treinamento de um `DecisionTreeClassifier`
- Avaliação com `accuracy_score`
- Experimentos com `test_size` e `random_state`

### Pré-Aula 02 — Análise Exploratória e Correlação
Aprofundamento em EDA: estatísticas descritivas, correlação, `groupby`, `regplot` e `boxplot`. Ênfase na distinção entre correlação e causalidade.

### Aula 02 — Pipeline com Dataset Titanic
Construção de um pipeline reprodutível de pré-processamento e modelagem aplicado ao clássico dataset do Titanic.

---

## Projeto Final — Agente de Saúde (Triagem para Diabetes)

O `agente-saude` é o projeto final da disciplina. Trata-se de uma aplicação fullstack que utiliza um modelo de Machine Learning treinado em Python para realizar a **triagem preditiva de diabetes** com base em dados clínicos do paciente.

### Como funciona

O sistema é composto por três camadas:

**1. Modelo Python (Google Colab / FastAPI)**
Modelo de classificação treinado com o [dataset Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). O notebook treina, avalia e expõe o modelo via FastAPI com `ngrok`, disponibilizando os endpoints:
- `GET /status` — verifica se o modelo está disponível e exibe métricas do último treino
- `POST /prever` — recebe dados clínicos e retorna diagnóstico com probabilidade de diabetes
- `GET /historico` — retorna o histórico de versões do modelo com métricas comparativas

**2. API Node.js (Fastify + TypeScript)**
BFF (Backend for Frontend) que atua como intermediário entre a interface web e o agente Python. Recebe as requisições do front, repassa ao Colab via URL do ngrok e trata erros de conectividade.

**3. Interface Web (React + Vite + Tailwind)**
Formulário clínico onde o usuário informa os dados do paciente (glicose, IMC, pressão arterial, histórico familiar, etc.) e recebe o diagnóstico em tempo real, incluindo probabilidade e nível de risco.

### Dados clínicos utilizados na predição

| Campo | Descrição |
|---|---|
| `gestacoes` | Número de gestações |
| `glicose` | Glicose plasmática (mg/dL) |
| `pressao_arterial` | Pressão arterial diastólica (mmHg) |
| `espessura_pele` | Dobra cutânea do tríceps (mm) |
| `insulina` | Insulina sérica (mu U/ml) |
| `imc` | Índice de Massa Corporal (kg/m²) |
| `historico_familiar` | Função de histórico familiar de diabetes |
| `idade` | Idade em anos |

### Arquitetura

```
[Interface Web]  ──►  [API Node.js / Fastify]  ──►  [FastAPI Python / Colab + ngrok]
   React + Vite           BFF TypeScript               Modelo de ML (sklearn)
```

### Notebook do Agente

> Aqui ficará o link para o notebook com o código completo do modelo Python (treino, avaliação e API).

**Notebook:** [Adicionar link aqui]()

---

## Tecnologias

**Notebooks / ML**
- Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- Google Colab, FastAPI, ngrok

**Projeto Final (agente-saude)**
- Node.js, Fastify, TypeScript, NX Monorepo
- React, Vite, Tailwind CSS

---

## Professor

Marcelo Batista
