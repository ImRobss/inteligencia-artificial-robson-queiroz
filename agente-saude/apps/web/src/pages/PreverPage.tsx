import { useState } from "react";
import { usePrever } from "../hooks/useAgente";
import ResultadoCard from "../components/ResultadoCard";
import type { PacienteDto } from "@saude/shared-types";

const camposConfig = [
  {
    key: "gestacoes",
    label: "Gestações",
    min: 0,
    max: 20,
    step: 1,
    placeholder: "Ex: 2",
  },
  {
    key: "glicose",
    label: "Glicose (mg/dL)",
    min: 1,
    max: 500,
    step: 1,
    placeholder: "Ex: 120",
  },
  {
    key: "pressao_arterial",
    label: "Pressão Arterial (mmHg)",
    min: 0,
    max: 200,
    step: 1,
    placeholder: "Ex: 70",
  },
  {
    key: "espessura_pele",
    label: "Espessura da Pele (mm)",
    min: 0,
    max: 100,
    step: 1,
    placeholder: "Ex: 28",
  },
  {
    key: "insulina",
    label: "Insulina (mu U/ml)",
    min: 0,
    max: 1000,
    step: 1,
    placeholder: "Ex: 0",
  },
  {
    key: "imc",
    label: "IMC (kg/m²)",
    min: 1,
    max: 100,
    step: 0.1,
    placeholder: "Ex: 32.5",
  },
  {
    key: "historico_familiar",
    label: "Histórico Familiar",
    min: 0,
    max: 2.5,
    step: 0.01,
    placeholder: "Ex: 0.627",
  },
  {
    key: "idade",
    label: "Idade (anos)",
    min: 1,
    max: 120,
    step: 1,
    placeholder: "Ex: 45",
  },
] as const;

const valoresIniciais: PacienteDto = {
  gestacoes: 2,
  glicose: 120,
  pressao_arterial: 70,
  espessura_pele: 20,
  insulina: 0,
  imc: 28.0,
  historico_familiar: 0.3,
  idade: 35,
};

export default function PreverPage() {
  const [form, setForm] = useState<PacienteDto>(valoresIniciais);
  const { resultado, loading, erro, prever } = usePrever();

  const handleChange = (key: keyof PacienteDto, value: string) => {
    setForm((f) => ({ ...f, [key]: parseFloat(value) || 0 }));
  };

  const handleSubmit = () => prever(form);

  return (
    <div className="space-y-8 max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-white">Nova Predição</h1>
        <p className="text-slate-400 text-sm mt-1">
          Informe os dados do paciente para análise de risco
        </p>
      </div>

      {/* Formulário */}
      <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 space-y-5">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          {camposConfig.map(({ key, label, min, max, step, placeholder }) => (
            <div key={key}>
              <label className="block text-sm font-medium text-slate-300 mb-1.5">
                {label}
              </label>
              <input
                type="number"
                min={min}
                max={max}
                step={step}
                value={form[key]}
                placeholder={placeholder}
                onChange={(e) => handleChange(key, e.target.value)}
                className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white
                           text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-brand-500
                           focus:border-transparent transition-all"
              />
            </div>
          ))}
        </div>

        <button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full bg-brand-600 hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed
                     text-white font-semibold py-2.5 rounded-lg transition-colors text-sm"
        >
          {loading ? "Analisando..." : "Analisar Paciente"}
        </button>
      </div>

      {/* Erro */}
      {erro && (
        <div className="bg-red-900/40 border border-red-700 rounded-xl p-4 text-red-300 text-sm">
          ⚠️ {erro}
        </div>
      )}

      {/* Resultado */}
      {resultado && <ResultadoCard resultado={resultado} />}
    </div>
  );
}
