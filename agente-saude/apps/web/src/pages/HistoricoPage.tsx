import { useHistorico } from "../hooks/useAgente";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function HistoricoPage() {
  const { historico, total, loading, recarregar } = useHistorico();

  if (loading)
    return (
      <div className="text-slate-400 animate-pulse text-sm">
        Carregando histórico...
      </div>
    );

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">
            Histórico de Treinamentos
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            {total} treino(s) registrado(s)
          </p>
        </div>
        <button
          onClick={recarregar}
          className="text-sm text-brand-400 hover:text-brand-300 transition-colors"
        >
          ↻ Atualizar
        </button>
      </div>

      {/* Gráfico de linhas */}
      {historico.length > 1 && (
        <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
          <h2 className="text-sm font-medium text-slate-300 mb-4">
            Evolução das Métricas
          </h2>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart
              data={historico.map((h, i) => ({ ...h, run: `#${i + 1}` }))}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="run" stroke="#94a3b8" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 1]} stroke="#94a3b8" tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: 8,
                }}
                labelStyle={{ color: "#f8fafc" }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="recall"
                stroke="#3b82f6"
                strokeWidth={2}
                dot
                name="Recall"
              />
              <Line
                type="monotone"
                dataKey="auc"
                stroke="#a855f7"
                strokeWidth={2}
                dot
                name="AUC"
              />
              <Line
                type="monotone"
                dataKey="f1"
                stroke="#22c55e"
                strokeWidth={2}
                dot
                name="F1"
              />
              <Line
                type="monotone"
                dataKey="acuracia"
                stroke="#f59e0b"
                strokeWidth={2}
                dot
                name="Acurácia"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Tabela */}
      <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-700/50 text-slate-300 text-left">
              <th className="px-4 py-3">#</th>
              <th className="px-4 py-3">Modelo</th>
              <th className="px-4 py-3">Recall</th>
              <th className="px-4 py-3">AUC</th>
              <th className="px-4 py-3">F1</th>
              <th className="px-4 py-3">Acurácia</th>
              <th className="px-4 py-3">Substituiu</th>
              <th className="px-4 py-3">Data</th>
            </tr>
          </thead>
          <tbody>
            {historico.length === 0 && (
              <tr>
                <td
                  colSpan={8}
                  className="px-4 py-8 text-center text-slate-500"
                >
                  Nenhum treino registrado ainda.
                </td>
              </tr>
            )}
            {[...historico].reverse().map((h, i) => (
              <tr
                key={i}
                className="border-t border-slate-700 hover:bg-slate-700/30 transition-colors"
              >
                <td className="px-4 py-3 text-slate-400">
                  {historico.length - i}
                </td>
                <td className="px-4 py-3 font-medium text-white">{h.modelo}</td>
                <td className="px-4 py-3 text-blue-400">
                  {(h.recall * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-3 text-purple-400">
                  {(h.auc * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-3 text-green-400">
                  {(h.f1 * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-3 text-amber-400">
                  {(h.acuracia * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-3">
                  {h.substituiu ? (
                    <span className="text-green-400">✓ Sim</span>
                  ) : (
                    <span className="text-slate-500">— Não</span>
                  )}
                </td>
                <td className="px-4 py-3 text-slate-400 text-xs">
                  {h.timestamp ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
