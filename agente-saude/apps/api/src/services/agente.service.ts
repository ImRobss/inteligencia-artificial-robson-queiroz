import { request } from "undici";
import { PacienteDto, RespostaPreditiva } from "../schemas/paciente.schema.js";
import { AgenteIndisponivelError } from "../controllers/saude.controller.js";

export class AgenteService {
  constructor(private readonly baseUrl: string) {}

  async status(): Promise<unknown> {
    try {
      return await this.get("/status");
    } catch {
      return { modelo_disponivel: false, mensagem: "Agente inacessível." };
    }
  }

  async historico(): Promise<unknown> {
    try {
      return await this.get("/historico");
    } catch {
      return { historico: [], total: 0 };
    }
  }

  async prever(dados: PacienteDto): Promise<RespostaPreditiva> {
    return this.post("/prever", dados) as Promise<RespostaPreditiva>;
  }

  private async get(path: string): Promise<unknown> {
    const { statusCode, body } = await request(`${this.baseUrl}${path}`, {
      headersTimeout: 10000,
      bodyTimeout: 10000,
    }).catch(() => {
      throw new AgenteIndisponivelError("Conexão recusada");
    });

    const text = await body.text();
    if (!text) throw new AgenteIndisponivelError("Resposta vazia do agente");

    const json = JSON.parse(text);
    if (statusCode !== 200) throw new AgenteIndisponivelError(String(json));
    return json;
  }

  private async post(path: string, payload: unknown): Promise<unknown> {
    const { statusCode, body } = await request(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      headersTimeout: 10000,
      bodyTimeout: 10000,
    }).catch(() => {
      throw new AgenteIndisponivelError("Conexão recusada");
    });

    const text = await body.text();
    if (!text) throw new AgenteIndisponivelError("Resposta vazia do agente");

    const json = JSON.parse(text);
    if (statusCode !== 200) throw new AgenteIndisponivelError(String(json));
    return json;
  }
}
