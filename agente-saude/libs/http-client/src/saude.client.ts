import type { PacienteDto, RespostaPreditiva } from '@saude/shared-types'

export type SaudeApiClient = typeof saudeApi

const BASE = typeof process !== 'undefined'
  ? (process.env['NEXT_PUBLIC_API_URL'] ?? 'http://localhost:3000')
  : (import.meta as { env?: { VITE_API_URL?: string } }).env?.VITE_API_URL ?? 'http://localhost:3000'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ erro: res.statusText }))
    throw new Error(err.erro ?? 'Erro desconhecido')
  }
  return res.json() as Promise<T>
}

export const saudeApi = {
  status: () =>
    fetch(`${BASE}/saude/status`).then(r => json<{ modelo_disponivel: boolean; mensagem: string }>(r)),

  prever: (dados: PacienteDto) =>
    fetch(`${BASE}/saude/prever`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(dados),
    }).then(r => json<RespostaPreditiva>(r)),

  historico: () =>
    fetch(`${BASE}/saude/historico`).then(r => json<{ historico: unknown[]; total: number }>(r)),
}
