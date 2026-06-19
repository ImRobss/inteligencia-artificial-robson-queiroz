import { FastifyInstance } from 'fastify'
import { SaudeController } from '../controllers/saude.controller.js'
import { pacienteSchema, respostaSchema } from '../schemas/paciente.schema.js'

export async function saudeRoutes(app: FastifyInstance) {
  const controller = new SaudeController()

  app.get('/status', {
    schema: {
      summary: 'Status do agente Python',
      tags: ['Saúde'],
    },
    handler: controller.status.bind(controller),
  })

  app.post('/prever', {
    schema: {
      summary: 'Predição de risco de diabetes',
      tags: ['Saúde'],
      body: pacienteSchema,
      response: { 200: respostaSchema },
    },
    handler: controller.prever.bind(controller),
  })

  app.get('/historico', {
    schema: {
      summary: 'Histórico de treinamentos',
      tags: ['Saúde'],
    },
    handler: controller.historico.bind(controller),
  })
}
