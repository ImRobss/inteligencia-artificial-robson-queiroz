import { buildApp } from '../../app.js'

// Mock do AgenteService para não depender do Colab nos testes
jest.mock('../../services/agente.service.js', () => ({
  AgenteService: jest.fn().mockImplementation(() => ({
    status:    jest.fn().mockResolvedValue({ modelo_disponivel: true, mensagem: 'Pronto.' }),
    historico: jest.fn().mockResolvedValue({ historico: [], total: 0 }),
    prever:    jest.fn().mockResolvedValue({
      diagnostico: 'Saudável', codigo: 0,
      probabilidade_diabetes: 0.12, risco: 'Baixo',
    }),
  })),
}))

// Seta env mínimo para o controller não lançar erro
process.env['FASTAPI_URL'] = 'http://mock-fastapi'

describe('Rotas /saude', () => {
  let app: Awaited<ReturnType<typeof buildApp>>

  beforeAll(async () => { app = await buildApp() })
  afterAll(async  () => { await app.close() })

  it('GET /saude/status → 200', async () => {
    const res = await app.inject({ method: 'GET', url: '/saude/status' })
    expect(res.statusCode).toBe(200)
    expect(res.json()).toHaveProperty('modelo_disponivel')
  })

  it('POST /saude/prever com body válido → 200', async () => {
    const body = {
      gestacoes:2, glicose:138, pressao_arterial:70,
      espessura_pele:28, insulina:0, imc:32.5,
      historico_familiar:0.3, idade:35,
    }
    const res = await app.inject({ method:'POST', url:'/saude/prever', payload: body })
    expect(res.statusCode).toBe(200)
    expect(res.json()).toHaveProperty('diagnostico')
  })

  it('POST /saude/prever com body inválido → 400', async () => {
    const res = await app.inject({ method:'POST', url:'/saude/prever', payload: { glicose: -1 } })
    expect(res.statusCode).toBe(400)
  })

  it('GET /saude/historico → 200', async () => {
    const res = await app.inject({ method: 'GET', url: '/saude/historico' })
    expect(res.statusCode).toBe(200)
  })
})
