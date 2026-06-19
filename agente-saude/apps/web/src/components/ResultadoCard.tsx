import type { RespostaPreditiva } from "@saude/shared-types";

function IconCheck() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  );
}

function IconWarn() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
    </svg>
  );
}

function IconAlert() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}

const riscoConfig = {
  Baixo: {
    text:   "text-green-400",
    border: "border-green-500/25",
    bg:     "bg-green-500/5",
    bar:    "bg-gradient-to-r from-green-700 to-green-400",
    icon:   <IconCheck />,
  },
  Médio: {
    text:   "text-amber-400",
    border: "border-amber-500/25",
    bg:     "bg-amber-500/5",
    bar:    "bg-gradient-to-r from-amber-700 to-amber-400",
    icon:   <IconWarn />,
  },
  Alto: {
    text:   "text-red-400",
    border: "border-red-500/25",
    bg:     "bg-red-500/5",
    bar:    "bg-gradient-to-r from-red-700 to-red-400",
    icon:   <IconAlert />,
  },
};

export default function ResultadoCard({ resultado }: { resultado: RespostaPreditiva }) {
  const cfg =
    riscoConfig[resultado.risco as keyof typeof riscoConfig] ?? riscoConfig["Médio"];
  const pct = (resultado.probabilidade_diabetes * 100).toFixed(1);

  return (
    <div className={`rounded-2xl p-6 border ${cfg.border} ${cfg.bg} space-y-5 animate-slide-up`}>
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className={`p-3 rounded-xl border ${cfg.border} ${cfg.text}`}>
          {cfg.icon}
        </div>
        <div>
          <p className="section-label mb-1">Resultado da Análise</p>
          <p className={`text-2xl font-bold tracking-tight ${cfg.text}`}>
            {resultado.diagnostico}
          </p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/40">
          <p className="section-label mb-2">Probabilidade</p>
          <p className={`text-2xl font-bold tracking-tight ${cfg.text}`}>{pct}%</p>
        </div>
        <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/40">
          <p className="section-label mb-2">Nível de Risco</p>
          <p className={`text-2xl font-bold tracking-tight ${cfg.text}`}>{resultado.risco}</p>
        </div>
      </div>

      {/* Probability bar */}
      <div>
        <div className="flex justify-between text-xs mb-2">
          <span className="text-slate-500">0%</span>
          <span className={`font-semibold ${cfg.text}`}>{pct}%</span>
          <span className="text-slate-500">100%</span>
        </div>
        <div className="w-full bg-slate-700/40 rounded-full h-3 overflow-hidden">
          <div
            className={`h-3 rounded-full ${cfg.bar} transition-all duration-1000 ease-out`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="flex justify-between mt-2 text-xs">
          <span className="text-green-600/70">Baixo</span>
          <span className="text-amber-600/70">Médio</span>
          <span className="text-red-600/70">Alto</span>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-3 bg-slate-900/40 border border-slate-700/30 rounded-xl p-3.5">
        <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 text-slate-500 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
          <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p className="text-xs text-slate-500 leading-relaxed">
          Este resultado é gerado por um modelo preditivo e não substitui o diagnóstico médico.
          Consulte sempre um profissional de saúde.
        </p>
      </div>
    </div>
  );
}
