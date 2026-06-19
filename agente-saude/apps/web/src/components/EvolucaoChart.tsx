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

export default function EvolucaoChart({ historico }: { historico: HistoricoItem[] }) {
  const data = historico.map((h, i) => ({
    run: `#${i + 1}`,
    Recall:    +(h.recall   * 100).toFixed(1),
    AUC:       +(h.auc      * 100).toFixed(1),
    F1:        +(h.f1       * 100).toFixed(1),
    Acurácia:  +(h.acuracia * 100).toFixed(1),
  }));

  return (
    <div className="card p-6">
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="run"
            stroke="#475569"
            tick={{ fontSize: 12, fill: "#94a3b8" }}
            axisLine={{ stroke: "#1e293b" }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            stroke="#475569"
            tick={{ fontSize: 12, fill: "#94a3b8" }}
            axisLine={false}
            tickLine={false}
            unit="%"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#0f172a",
              border: "1px solid #1e293b",
              borderRadius: 12,
              fontSize: 13,
            }}
            labelStyle={{ color: "#f8fafc", fontWeight: 600 }}
            formatter={(v: number) => [`${v}%`]}
          />
          <Legend
            wrapperStyle={{ fontSize: 13, paddingTop: 12 }}
            iconType="circle"
            iconSize={8}
          />
          <ReferenceLine
            y={72}
            stroke="#ef4444"
            strokeDasharray="4 2"
            strokeOpacity={0.5}
            label={{ value: "Limiar", fill: "#ef4444", fontSize: 11, opacity: 0.7 }}
          />
          <Line type="monotone" dataKey="Recall"   stroke="#3b82f6" strokeWidth={2} dot={{ r: 3, fill: "#3b82f6" }} />
          <Line type="monotone" dataKey="AUC"      stroke="#a855f7" strokeWidth={2} dot={{ r: 3, fill: "#a855f7" }} />
          <Line type="monotone" dataKey="F1"       stroke="#22c55e" strokeWidth={2} dot={{ r: 3, fill: "#22c55e" }} />
          <Line type="monotone" dataKey="Acurácia" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3, fill: "#f59e0b" }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
