/**
 * app.ts — Factory do servidor Fastify.
 * Separado do main.ts para facilitar testes de integração.
 */
import Fastify, { FastifyInstance } from 'fastify'
import cors from '@fastify/cors'
import { saudeRoutes } from './routes/saude.routes.js'

export async function buildApp(): Promise<FastifyInstance> {
  const app = Fastify({
    logger: {
      transport: {
        target: 'pino-pretty',
        options: { colorize: true },
      },
    },
  })

  // Plugins globais
  await app.register(cors, { origin: '*', methods: ['GET', 'POST'] })

  // Health check raiz
  app.get('/', async () => ({
    status:  'online',
    servico: 'Agente Saúde API',
    versao:  '2.0.0',
    rotas: [
      'GET  /saude/status',
      'POST /saude/prever',
      'GET  /saude/historico',
    ],
  }))

  // Módulos de rotas
  await app.register(saudeRoutes, { prefix: '/saude' })

  return app
}
