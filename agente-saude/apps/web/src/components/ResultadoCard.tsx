import type { RespostaPreditiva } from "@saude/shared-types";

const riscoConfig = {
  Baixo: {
    color: "text-green-400",
    bg: "bg-green-900/30  border-green-700",
    emoji: "✅",
  },
  Médio: {
    color: "text-amber-400",
    bg: "bg-amber-900/30  border-amber-700",
    emoji: "⚠️",
  },
  Alto: {
    color: "text-red-400",
    bg: "bg-red-900/30    border-red-700",
    emoji: "🚨",
  },
};

export default function ResultadoCard({
  resultado,
}: {
  resultado: RespostaPreditiva;
}) {
  const cfg =
    riscoConfig[resultado.risco as keyof typeof riscoConfig] ??
    riscoConfig["Médio"];
  const pct = (resultado.probabilidade_diabetes * 100).toFixed(1);

  return (
    <div className={`rounded-xl p-6 border ${cfg.bg} space-y-4`}>
      <div className="flex items-center gap-3">
        <span className="text-3xl">{cfg.emoji}</span>
        <div>
          <p className="text-xs text-slate-400 uppercase tracking-wider">
            Diagnóstico
          </p>
          <p className={`text-2xl font-bold ${cfg.color}`}>
            {resultado.diagnostico}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-slate-800/60 rounded-lg p-3">
          <p className="text-xs text-slate-400 mb-1">Probabilidade</p>
          <p className={`text-xl font-bold ${cfg.color}`}>{pct}%</p>
        </div>
        <div className="bg-slate-800/60 rounded-lg p-3">
          <p className="text-xs text-slate-400 mb-1">Nível de Risco</p>
          <p className={`text-xl font-bold ${cfg.color}`}>{resultado.risco}</p>
        </div>
      </div>

      {/* Barra de probabilidade */}
      <div>
        <div className="flex justify-between text-xs text-slate-400 mb-1">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
        <div className="w-full bg-slate-700 rounded-full h-2.5">
          <div
            className={`h-2.5 rounded-full transition-all duration-700 ${
              resultado.risco === "Alto"
                ? "bg-red-500"
                : resultado.risco === "Médio"
                  ? "bg-amber-500"
                  : "bg-green-500"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      <p className="text-xs text-slate-500">
        ⚕️ Este resultado é gerado por um modelo preditivo. Consulte sempre um
        profissional de saúde.
      </p>
    </div>
  );
}
