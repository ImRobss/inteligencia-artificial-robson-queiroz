# Pré-Aula 02 — Lembretes de Código e Funcionalidades

## Imports usados

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
```

---

## Criar DataFrame sintético

```python
np.random.seed(42)  # garante reprodutibilidade

df = pd.DataFrame({
    "student_id": np.arange(1, n+1),
    "gender": np.random.choice(["female", "male"], size=n, p=[0.55, 0.45]),
    "age": np.random.randint(17, 31, size=n),
    "study_hours": np.round(np.random.gamma(shape=2.0, scale=2.0, size=n), 1),
    "absences": np.random.poisson(lam=2.0, size=n),
})
```

**Lembrete:** `np.random.gamma()` e `np.random.poisson()` geram distribuições diferentes — gamma para valores contínuos, poisson para contagens discretas.

---

## Salvar e ler CSV

```python
df.to_csv("students_simple.csv", index=False)  # salva sem índice
df = pd.read_csv("students_simple.csv")         # lê de volta
```

---

## Checar estrutura do dataset

```python
df.shape          # (linhas, colunas)
df.info()         # tipos e não-nulos
df.columns        # lista de colunas
df.head(10)       # primeiras 10 linhas
```

---

## Verificar valores faltantes

```python
missing = df.isna().sum().sort_values(ascending=False)
faltantes = missing[missing > 0]

if len(faltantes) == 0:
    print("✅ Não há valores faltantes.")
else:
    display(faltantes)
```

**Lembrete:** `isna()` retorna booleano; `sum()` conta `True` por coluna.

---

## Estatísticas descritivas

```python
df.describe(include="all").T   # transposto para facilitar leitura
```

### Medidas específicas

```python
df["final_score"].mean()    # média
df["final_score"].median()  # mediana
df["final_score"].std()     # desvio padrão
```

---

## Correlação

```python
df[["study_hours", "final_score"]].corr()
# ou valor direto:
corr = df[["study_hours", "final_score"]].corr().loc["study_hours", "final_score"]
```

**Lembrete:**

- Correlação perto de **+1** → associação positiva forte
- Perto de **-1** → associação negativa forte
- Perto de **0** → sem associação linear
- **Correlação ≠ causalidade**

---

## Groupby — agregação por grupo

```python
summary = (
    df.groupby("gender")
      .agg(
          n=("final_score", "size"),
          mean_score=("final_score", "mean"),
          median_score=("final_score", "median"),
          std_score=("final_score", "std"),
          mean_absences=("absences", "mean"),
          mean_study_hours=("study_hours", "mean"),
      )
      .sort_values("mean_score", ascending=False)
)
```

**Lembrete:** `.agg()` aceita tuplas `(coluna, função)` para criar colunas nomeadas.

---

## Gráficos

### Scatter com linha de regressão

```python
sns.regplot(data=df, x="study_hours", y="final_score", scatter_kws={"alpha": 0.6})
plt.title("Relação entre horas de estudo e nota final")
plt.xlabel("Horas de estudo (semana)")
plt.ylabel("Nota final")
plt.show()
```

**Lembrete:** `regplot` desenha scatter + linha de regressão automaticamente.

### Boxplot por grupo

```python
sns.boxplot(data=df, x="gender", y="final_score")
plt.title("Distribuição de nota final por gênero")
plt.xlabel("Gênero")
plt.ylabel("Nota final")
plt.show()
```

**Lembrete:** Boxplot mostra mediana (linha central), quartis (caixa) e outliers (pontos fora dos whiskers).

---

## Regra de ouro

> **Correlação NÃO prova causalidade.**  
> Dizer "há associação positiva entre X e Y" ≠ dizer "X causa Y".

---

## Fluxo de trabalho

1. Carregar CSV
2. Checar shape, info, missing
3. Estatísticas descritivas (describe, mean, std)
4. Correlação entre variáveis numéricas
5. Groupby para comparar grupos
6. Gráficos (scatter, boxplot)
7. Interpretar de forma objetiva (sem inventar causalidade)
