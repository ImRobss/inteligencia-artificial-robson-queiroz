import { useState, useEffect, useCallback } from "react";
import type { PacienteDto, RespostaPreditiva } from "@saude/shared-types";

const API = "/saude";

export type HistoricoItem = {
  timestamp: string;
  modelo: string;
  recall: number;
  auc: number;
  f1: number;
  acuracia: number;
  precisao: number;
  substituiu: boolean;
};

export type StatusAgente = {
  modelo_disponivel: boolean;
  ultimo_treino?: HistoricoItem;
};

async function safeFetch<T>(
  url: string,
  fallback: T,
  options?: RequestInit,
): Promise<T> {
  try {
    const res = await fetch(url, options);
    const text = await res.text();
    if (!text) return fallback;
    return JSON.parse(text) as T;
  } catch {
    return fallback;
  }
}

export function useStatus() {
  const [data, setData] = useState<StatusAgente>({ modelo_disponivel: false });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    safeFetch<StatusAgente>(`${API}/status`, { modelo_disponivel: false })
      .then(setData)
      .finally(() => setLoading(false));
  }, []);

  return { data, loading };
}

export function useHistorico() {
  const [historico, setHistorico] = useState<HistoricoItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);

  const carregar = useCallback(() => {
    setLoading(true);
    safeFetch<{ historico: HistoricoItem[]; total: number }>(
      `${API}/historico`,
      { historico: [], total: 0 },
    )
      .then((d) => {
        setHistorico(d.historico ?? []);
        setTotal(d.total ?? 0);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    carregar();
  }, [carregar]);

  return { historico, total, loading, recarregar: carregar };
}

export function usePrever() {
  const [resultado, setResultado] = useState<RespostaPreditiva | null>(null);
  const [loading, setLoading] = useState(false);
  const [erro, setErro] = useState<string | null>(null);

  const prever = useCallback(async (dados: PacienteDto) => {
    setLoading(true);
    setErro(null);
    setResultado(null);
    try {
      const res = await fetch(`${API}/prever`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(dados),
      });
      const text = await res.text();
      if (!text) throw new Error("Resposta vazia do servidor");
      const json = JSON.parse(text);
      if (!res.ok) throw new Error(json.erro ?? "Erro na predição");
      setResultado(json);
    } catch (e) {
      setErro((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  return { resultado, loading, erro, prever };
}
