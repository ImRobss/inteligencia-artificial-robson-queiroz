import { useStatus, useHistorico } from "../hooks/useAgente";
import MetricCard from "../components/MetricCard";
import EvolucaoChart from "../components/EvolucaoChart";
import StatusBadge from "../components/StatusBadge";

export default function DashboardPage() {
  const { data: status, loading: loadingStatus } = useStatus();
  const { historico = [], total, loading: loadingHist } = useHistorico();

  const ultimo = historico.length > 0 ? historico[historico.length - 1] : null;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-slate-400 text-sm mt-1">
          Visão geral do agente de diagnóstico
        </p>
      </div>

      <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 flex items-center gap-4">
        {loadingStatus ? (
          <div className="text-slate-400 text-sm animate-pulse">
            Verificando agente...
          </div>
        ) : (
          <>
            <StatusBadge disponivel={status?.modelo_disponivel ?? false} />
            <div>
              <p className="text-sm font-medium text-white">
                {status?.modelo_disponivel
                  ? "Modelo disponível para predições"
                  : "Modelo não treinado"}
              </p>
              {ultimo && (
                <p className="text-xs text-slate-400 mt-0.5">
                  Último treino: {ultimo.modelo} — {ultimo.timestamp}
                </p>
              )}
            </div>
          </>
        )}
      </div>

      {ultimo && (
        <div>
          <h2 className="text-lg font-semibold text-white mb-4">
            Métricas do Modelo Atual
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard label="Recall" value={ultimo.recall} color="blue" />
            <MetricCard label="AUC-ROC" value={ultimo.auc} color="purple" />
            <MetricCard label="F1-Score" value={ultimo.f1} color="green" />
            <MetricCard
              label="Acurácia"
              value={ultimo.acuracia}
              color="amber"
            />
          </div>
        </div>
      )}

      {!loadingHist && historico.length > 1 && (
        <div>
          <h2 className="text-lg font-semibold text-white mb-4">
            Evolução do Agente
            <span className="ml-2 text-sm font-normal text-slate-400">
              ({total} treinos)
            </span>
          </h2>
          <EvolucaoChart historico={historico} />
        </div>
      )}

      {!loadingHist && historico.length === 0 && (
        <div className="bg-slate-800 rounded-xl p-8 border border-slate-700 text-center text-slate-400">
          Agente offline ou sem treinos registrados. Verifique a conexão com a
          API.
        </div>
      )}
    </div>
  );
}
