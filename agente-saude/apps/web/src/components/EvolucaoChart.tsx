import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { HistoricoItem } from "../hooks/useAgente";

export default function EvolucaoChart({
  historico,
}: {
  historico: HistoricoItem[];
}) {
  const data = historico.map((h, i) => ({
    run: `#${i + 1}`,
    Recall: +(h.recall * 100).toFixed(1),
    AUC: +(h.auc * 100).toFixed(1),
    F1: +(h.f1 * 100).toFixed(1),
    Acurácia: +(h.acuracia * 100).toFixed(1),
  }));

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700">
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="run" stroke="#94a3b8" tick={{ fontSize: 12 }} />
          <YAxis
            domain={[0, 100]}
            stroke="#94a3b8"
            tick={{ fontSize: 12 }}
            unit="%"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1e293b",
              border: "1px solid #334155",
              borderRadius: 8,
            }}
            labelStyle={{ color: "#f8fafc" }}
            formatter={(v: number) => `${v}%`}
          />
          <Legend />
          <ReferenceLine
            y={72}
            stroke="#ef4444"
            strokeDasharray="4 2"
            label={{ value: "Limiar", fill: "#ef4444", fontSize: 11 }}
          />
          <Line
            type="monotone"
            dataKey="Recall"
            stroke="#3b82f6"
            strokeWidth={2}
            dot
          />
          <Line
            type="monotone"
            dataKey="AUC"
            stroke="#a855f7"
            strokeWidth={2}
            dot
          />
          <Line
            type="monotone"
            dataKey="F1"
            stroke="#22c55e"
            strokeWidth={2}
            dot
          />
          <Line
            type="monotone"
            dataKey="Acurácia"
            stroke="#f59e0b"
            strokeWidth={2}
            dot
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
