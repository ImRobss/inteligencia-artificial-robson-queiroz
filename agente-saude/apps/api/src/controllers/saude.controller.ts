import { FastifyRequest, FastifyReply } from 'fastify'
import { AgenteService } from '../services/agente.service.js'
import { PacienteDto } from '../schemas/paciente.schema.js'

export class SaudeController {
  private readonly agente: AgenteService

  constructor() {
    const fastapiUrl = process.env['FASTAPI_URL']
    if (!fastapiUrl) throw new Error('FASTAPI_URL não definida no .env')
    this.agente = new AgenteService(fastapiUrl)
  }

  async status(_req: FastifyRequest, reply: FastifyReply) {
    try {
      const data = await this.agente.status()
      return reply.send(data)
    } catch (err) {
      return this.handleError(err, reply)
    }
  }

  async prever(req: FastifyRequest<{ Body: PacienteDto }>, reply: FastifyReply) {
    try {
      const resultado = await this.agente.prever(req.body)
      return reply.send(resultado)
    } catch (err) {
      return this.handleError(err, reply)
    }
  }

  async historico(_req: FastifyRequest, reply: FastifyReply) {
    try {
      const data = await this.agente.historico()
      return reply.send(data)
    } catch (err) {
      return this.handleError(err, reply)
    }
  }

  private handleError(err: unknown, reply: FastifyReply) {
    if (err instanceof AgenteIndisponivelError) {
      return reply.status(502).send({
        erro: 'Agente Python inacessível.',
        detalhe: 'Verifique se o Colab está rodando e a URL do ngrok está correta.',
      })
    }
    return reply.status(500).send({ erro: 'Erro interno do servidor.' })
  }
}

export class AgenteIndisponivelError extends Error {
  constructor(msg: string) {
    super(msg)
    this.name = 'AgenteIndisponivelError'
  }
}
