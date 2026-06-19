/**
 * main.ts — Bootstrap da API
 * Carrega .env, registra plugins e sobe o servidor Fastify.
 */
import 'dotenv/config'
import { buildApp } from './app.js'

const start = async () => {
  const app = await buildApp()

  try {
    const port = Number(process.env['PORT'] ?? 3000)
    await app.listen({ port, host: '0.0.0.0' })
    app.log.info(`FastAPI alvo: ${process.env['FASTAPI_URL']}`)
  } catch (err) {
    app.log.error(err)
    process.exit(1)
  }
}

start()
