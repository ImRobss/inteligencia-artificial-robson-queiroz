import { useHistorico } from "../hooks/useAgente";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";

function SkeletonTable() {
  return (
    <div className="card overflow-hidden animate-pulse">
      <div className="h-10 bg-slate-700/40 border-b border-slate-700/60" />
      {[...Array(5)].map((_, i) => (
        <div key={i} className="flex gap-4 px-4 py-3.5 border-b border-slate-700/30">
          <div className="w-6 h-3.5 bg-slate-700/60 rounded" />
          <div className="w-32 h-3.5 bg-slate-700/60 rounded" />
          <div className="w-14 h-3.5 bg-slate-700/40 rounded" />
          <div className="w-14 h-3.5 bg-slate-700/40 rounded" />
          <div className="w-14 h-3.5 bg-slate-700/40 rounded" />
          <div className="w-14 h-3.5 bg-slate-700/40 rounded" />
        </div>
      ))}
    </div>
  );
}

export default function HistoricoPage() {
  const { historico, total, loading, recarregar } = useHistorico();

  if (loading) return (
    <div className="space-y-8">
      <div className="h-8 w-64 bg-slate-700/60 rounded-lg animate-pulse" />
      <SkeletonTable />
    </div>
  );

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">Histórico de Treinamentos</h1>
          <p className="text-slate-400 text-sm mt-1.5">{total} treino(s) registrado(s)</p>
        </div>
        <button
          onClick={recarregar}
          className="flex items-center gap-1.5 text-sm font-medium text-slate-400
                     hover:text-slate-200 border border-slate-700/60 px-3.5 py-2 rounded-xl
                     hover:bg-slate-800 transition-all duration-200"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Atualizar
        </button>
      </div>

      {/* Chart */}
      {historico.length > 1 && (
        <div className="card p-6">
          <p className="text-sm font-semibold text-slate-300 mb-5">Evolução das Métricas</p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={historico.map((h, i) => ({
              run: `#${i + 1}`,
              Recall:   +(h.recall   * 100).toFixed(1),
              AUC:      +(h.auc      * 100).toFixed(1),
              F1:       +(h.f1       * 100).toFixed(1),
              Acurácia: +(h.acuracia * 100).toFixed(1),
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="run" stroke="#475569" tick={{ fontSize: 12, fill: "#94a3b8" }} axisLine={{ stroke: "#1e293b" }} tickLine={false} />
              <YAxis domain={[0, 100]} stroke="#475569" tick={{ fontSize: 12, fill: "#94a3b8" }} axisLine={false} tickLine={false} unit="%" />
              <Tooltip
                contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #1e293b", borderRadius: 12, fontSize: 13 }}
                labelStyle={{ color: "#f8fafc", fontWeight: 600 }}
                formatter={(v: number) => [`${v}%`]}
              />
              <Legend wrapperStyle={{ fontSize: 13, paddingTop: 12 }} iconType="circle" iconSize={8} />
              <Line type="monotone" dataKey="Recall"   stroke="#3b82f6" strokeWidth={2} dot={{ r: 3, fill: "#3b82f6" }} />
              <Line type="monotone" dataKey="AUC"      stroke="#a855f7" strokeWidth={2} dot={{ r: 3, fill: "#a855f7" }} />
              <Line type="monotone" dataKey="F1"       stroke="#22c55e" strokeWidth={2} dot={{ r: 3, fill: "#22c55e" }} />
              <Line type="monotone" dataKey="Acurácia" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3, fill: "#f59e0b" }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Table */}
      <div className="card overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700/60">
              <th className="px-5 py-3.5 text-left section-label">#</th>
              <th className="px-5 py-3.5 text-left section-label">Modelo</th>
              <th className="px-5 py-3.5 text-left section-label">Recall</th>
              <th className="px-5 py-3.5 text-left section-label">AUC</th>
              <th className="px-5 py-3.5 text-left section-label">F1</th>
              <th className="px-5 py-3.5 text-left section-label">Acurácia</th>
              <th className="px-5 py-3.5 text-left section-label">Substituiu</th>
              <th className="px-5 py-3.5 text-left section-label">Data</th>
            </tr>
          </thead>
          <tbody>
            {historico.length === 0 && (
              <tr>
                <td colSpan={8} className="px-5 py-12 text-center">
                  <div className="w-10 h-10 rounded-xl bg-slate-700/30 flex items-center justify-center mx-auto mb-3">
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm font-medium">Nenhum treino registrado</p>
                  <p className="text-slate-600 text-xs mt-1">Os registros aparecerão aqui após o primeiro treino</p>
                </td>
              </tr>
            )}
            {[...historico].reverse().map((h, i) => (
              <tr
                key={i}
                className="border-t border-slate-700/30 hover:bg-slate-700/20 transition-colors duration-150"
              >
                <td className="px-5 py-3.5 text-slate-500 text-xs font-mono">
                  {historico.length - i}
                </td>
                <td className="px-5 py-3.5">
                  <span className="font-medium text-slate-200">{h.modelo}</span>
                </td>
                <td className="px-5 py-3.5 text-blue-400 font-medium">
                  {(h.recall * 100).toFixed(1)}%
                </td>
                <td className="px-5 py-3.5 text-purple-400 font-medium">
                  {(h.auc * 100).toFixed(1)}%
                </td>
                <td className="px-5 py-3.5 text-green-400 font-medium">
                  {(h.f1 * 100).toFixed(1)}%
                </td>
                <td className="px-5 py-3.5 text-amber-400 font-medium">
                  {(h.acuracia * 100).toFixed(1)}%
                </td>
                <td className="px-5 py-3.5">
                  {h.substituiu ? (
                    <span className="inline-flex items-center gap-1 text-xs font-semibold text-green-400 bg-green-500/10 border border-green-500/25 px-2.5 py-1 rounded-full">
                      <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
                      Sim
                    </span>
                  ) : (
                    <span className="text-slate-600 text-xs">—</span>
                  )}
                </td>
                <td className="px-5 py-3.5 text-slate-500 text-xs">
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
