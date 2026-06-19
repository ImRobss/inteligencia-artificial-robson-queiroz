# Agente Saúde — Monorepo NX

Sistema full-stack de diagnóstico preditivo de diabetes. O frontend React consome uma API Fastify em Node.js, que por sua vez repassa as requisições para um agente em Python (FastAPI) executando o modelo de machine learning.

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                        Usuário (Browser)                        │
│                      React SPA — Vite 5                         │
│              http://localhost:5173                               │
└────────────────────────────┬────────────────────────────────────┘
                             │ Proxy /saude → :3000
┌────────────────────────────▼────────────────────────────────────┐
│                      API Node.js — Fastify 4                    │
│                      http://localhost:3000                       │
│  GET  /                 → health check                          │
│  GET  /saude/status     → status do modelo Python               │
│  POST /saude/prever     → predição de diabetes                  │
│  GET  /saude/historico  → histórico de treinos                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP via undici (FASTAPI_URL)
┌────────────────────────────▼────────────────────────────────────┐
│              Agente Python — FastAPI (externo / ngrok)          │
│              Modelo de ML: predição de diabetes                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Estrutura de pastas

```
agente-saude/
│
├── apps/
│   ├── api/                        ← API Fastify (Node.js + TypeScript)
│   │   └── src/
│   │       ├── main.ts             ← bootstrap: carrega .env e inicia servidor
│   │       ├── app.ts              ← factory Fastify: CORS, health check, rotas
│   │       ├── routes/
│   │       │   ├── saude.routes.ts ← declara GET /status, POST /prever, GET /historico
│   │       │   └── __tests__/
│   │       │       └── saude.spec.ts
│   │       ├── controllers/
│   │       │   └── saude.controller.ts ← recebe request, chama service, devolve resposta
│   │       ├── services/
│   │       │   └── agente.service.ts   ← cliente HTTP para o agente Python (undici)
│   │       └── schemas/
│   │           └── paciente.schema.ts  ← JSON Schema para validação e tipagem
│   │
│   └── web/                        ← Frontend React + Vite + Tailwind
│       └── src/
│           ├── main.tsx            ← entry point React (BrowserRouter)
│           ├── App.tsx             ← roteamento e layout (navbar, dark theme)
│           ├── pages/
│           │   ├── DashboardPage.tsx   ← status do modelo + métricas + gráfico de evolução
│           │   ├── PreverPage.tsx      ← formulário de predição (8 campos do paciente)
│           │   └── HistoricoPage.tsx   ← tabela e gráfico do histórico de treinos
│           ├── components/
│           │   ├── EvolucaoChart.tsx   ← LineChart (recharts) com 4 métricas
│           │   ├── MetricCard.tsx      ← card de métrica individual
│           │   ├── ResultadoCard.tsx   ← exibe resultado da predição com barra de risco
│           │   └── StatusBadge.tsx     ← badge Online / Offline
│           ├── hooks/
│           │   └── useAgente.ts        ← hooks: useStatus, useHistorico, usePrever
│           └── styles/
│               └── globals.css
│
├── libs/
│   ├── shared-types/               ← tipos TypeScript compartilhados entre apps e libs
│   │   └── src/
│   │       ├── index.ts
│   │       ├── paciente.types.ts   ← PacienteDto, RespostaPreditiva
│   │       └── api.types.ts        ← ApiResponse<T>, ApiError
│   │
│   └── http-client/                ← cliente fetch reutilizável (saudeApi)
│       └── src/
│           ├── index.ts
│           └── saude.client.ts     ← métodos: status(), prever(), historico()
│
├── .env                            ← variáveis de ambiente (não versionado)
├── .env.example                    ← template de variáveis
├── nx.json                         ← configuração do workspace Nx
├── package.json                    ← scripts e dependências raiz
└── tsconfig.base.json              ← TypeScript base com path aliases
```

---

## Tech Stack

| Camada | Tecnologia |
|--------|-----------|
| Monorepo | Nx 19, npm workspaces |
| Linguagem | TypeScript 5.4 (strict) |
| API backend | Fastify 4.27, @fastify/cors, undici |
| Frontend | React 18, React Router 6, Vite 5 |
| UI | Tailwind CSS 3.4, Recharts 2.12 |
| Bundler de libs | tsup (gera CJS + ESM + .d.ts) |
| Testes | Jest 29, ts-jest, @nx/jest |
| Linting | ESLint 8, @typescript-eslint, @nx/eslint-plugin |

---

## Variáveis de ambiente

Crie um arquivo `.env` na raiz do monorepo copiando `.env.example`:

```bash
cp .env.example .env
```

| Variável | Descrição | Exemplo |
|----------|-----------|---------|
| `FASTAPI_URL` | URL base do agente Python (ngrok ou local) | `https://xxxx.ngrok.io` |
| `PORT` | Porta da API Fastify | `3000` |
| `NODE_ENV` | Ambiente de execução | `development` |

---

## Como rodar

### Pré-requisitos

- Node.js 20+
- O agente Python (FastAPI) rodando e acessível (local ou via ngrok)

### 1. Instalar dependências

```bash
npm install
```

### 2. Configurar o ambiente

```bash
cp .env.example .env
# Edite .env e preencha FASTAPI_URL com a URL do agente Python
```

### 3. Rodar a API em desenvolvimento (hot reload)

```bash
npm run api:dev
# Disponível em http://localhost:3000
```

### 4. Rodar o frontend em desenvolvimento

```bash
cd apps/web && npm run dev
# Disponível em http://localhost:5173
# O proxy /saude já aponta para http://localhost:3000
```

### 5. Rodar API e frontend juntos

Abra dois terminais e execute os dois comandos acima em paralelo.

---

## Scripts disponíveis

Executados na raiz do monorepo:

```bash
npm run api:dev      # API com hot reload (tsx watch)
npm run api:build    # Build de produção da API (dist/apps/api)
npm run api:test     # Testes unitários da API
npm run lint         # Lint em todos os packages
npm run test         # Testes em todos os packages
npm run graph        # Abre o grafo de dependências no browser
```

Executados em `apps/web/`:

```bash
npm run dev          # Frontend com HMR
npm run build        # Build de produção (dist/)
npm run preview      # Preview do build de produção
```

---

## Endpoints da API

### `GET /`

Health check.

```json
{ "status": "ok" }
```

### `GET /saude/status`

Verifica se o agente Python está disponível e o modelo carregado.

```json
{
  "model_disponivel": true
}
```

### `POST /saude/prever`

Envia dados do paciente e recebe a predição de risco de diabetes.

**Body:**

```json
{
  "gestacoes": 2,
  "glicose": 120,
  "pressao_arterial": 70,
  "espessura_pele": 20,
  "insulina": 80,
  "imc": 28.5,
  "historico_familiar": 0.5,
  "idade": 35
}
```

**Resposta:**

```json
{
  "diagnostico": "Sem diagnóstico de diabetes",
  "codigo": 0,
  "probabilidade_diabetes": 0.23,
  "risco": "baixo"
}
```

### `GET /saude/historico`

Retorna o histórico de execuções de treino do modelo.

```json
[
  {
    "data": "2026-06-19T10:00:00Z",
    "recall": 0.78,
    "auc": 0.84,
    "f1": 0.75,
    "acuracia": 0.82
  }
]
```

---

## Páginas do frontend

| Rota | Página | Descrição |
|------|--------|-----------|
| `/` | Dashboard | Status do modelo, métricas do último treino e gráfico de evolução |
| `/prever` | Nova Predição | Formulário com 8 campos do paciente; exibe resultado com barra de probabilidade |
| `/historico` | Histórico | Tabela e gráfico de linha com todas as execuções de treino |

---

## Path aliases TypeScript

Definidos em `tsconfig.base.json` e disponíveis em todos os apps e libs:

```ts
import type { PacienteDto } from '@saude/shared-types'
import { saudeApi } from '@saude/http-client'
```

---

## Regras de dependência (ESLint boundaries)

O Nx enforce as seguintes fronteiras via `@nx/eslint-plugin`:

- `scope:api` → só pode importar `scope:shared`
- `scope:web` → só pode importar `scope:shared`
- `scope:shared` → só pode importar `scope:shared`

---

## Build de produção

```bash
# Build da API
npm run api:build
# Saída: dist/apps/api/main.js

# Iniciar em produção
node dist/apps/api/main.js
```

```bash
# Build do frontend
cd apps/web && npm run build
# Saída: apps/web/dist/
# Sirva a pasta dist/ com qualquer servidor estático (nginx, serve, etc.)
```
