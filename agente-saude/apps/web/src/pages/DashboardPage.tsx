import { Link } from "react-router-dom";
import { useStatus, useHistorico } from "../hooks/useAgente";
import MetricCard from "../components/MetricCard";
import EvolucaoChart from "../components/EvolucaoChart";
import StatusBadge from "../components/StatusBadge";

function SkeletonLine({ w }: { w: string }) {
  return <div className={`h-3.5 ${w} bg-slate-700/60 rounded animate-pulse`} />;
}

export default function DashboardPage() {
  const { data: status, loading: loadingStatus } = useStatus();
  const { historico = [], total, loading: loadingHist } = useHistorico();

  const ultimo = historico.length > 0 ? historico[historico.length - 1] : null;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">Dashboard</h1>
        <p className="text-slate-400 text-sm mt-1.5">Visão geral do agente de diagnóstico</p>
      </div>

      {/* Status card */}
      <div className="card p-5">
        {loadingStatus ? (
          <div className="flex items-center gap-4">
            <div className="w-16 h-6 bg-slate-700/60 rounded-full animate-pulse" />
            <div className="space-y-2">
              <SkeletonLine w="w-48" />
              <SkeletonLine w="w-64" />
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <StatusBadge disponivel={status?.modelo_disponivel ?? false} />
              <div>
                <p className="text-sm font-medium text-white">
                  {status?.modelo_disponivel
                    ? "Modelo disponível para predições"
                    : "Modelo não treinado ou offline"}
                </p>
                {ultimo && (
                  <p className="text-xs text-slate-500 mt-0.5">
                    Último treino: {ultimo.modelo} · {ultimo.timestamp}
                  </p>
                )}
              </div>
            </div>
            {status?.modelo_disponivel && (
              <Link
                to="/prever"
                className="shrink-0 text-xs font-semibold text-brand-400 border border-brand-500/30
                           px-3.5 py-1.5 rounded-xl hover:bg-brand-500/10 hover:text-brand-300
                           transition-all duration-200"
              >
                Iniciar predição →
              </Link>
            )}
          </div>
        )}
      </div>

      {/* Metrics */}
      {!loadingHist && ultimo && (
        <div>
          <h2 className="text-lg font-semibold text-slate-200 tracking-tight mb-4">
            Métricas do Modelo Atual
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard label="Recall"   value={ultimo.recall}   color="blue"   />
            <MetricCard label="AUC-ROC"  value={ultimo.auc}      color="purple" />
            <MetricCard label="F1-Score" value={ultimo.f1}       color="green"  />
            <MetricCard label="Acurácia" value={ultimo.acuracia} color="amber"  />
          </div>
        </div>
      )}

      {/* Chart */}
      {!loadingHist && historico.length > 1 && (
        <div>
          <h2 className="text-lg font-semibold text-slate-200 tracking-tight mb-4">
            Evolução do Agente{" "}
            <span className="text-sm font-normal text-slate-500">({total} treinos)</span>
          </h2>
          <EvolucaoChart historico={historico} />
        </div>
      )}

      {/* Empty state */}
      {!loadingHist && historico.length === 0 && (
        <div className="card p-12 text-center">
          <div className="w-12 h-12 rounded-2xl bg-slate-700/40 flex items-center justify-center mx-auto mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <p className="text-slate-300 text-sm font-medium">Sem dados disponíveis</p>
          <p className="text-slate-600 text-xs mt-1">Verifique a conexão com a API do agente</p>
        </div>
      )}
    </div>
  );
}
