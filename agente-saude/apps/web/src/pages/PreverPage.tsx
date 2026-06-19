import { useState } from "react";
import { usePrever } from "../hooks/useAgente";
import ResultadoCard from "../components/ResultadoCard";
import type { PacienteDto } from "@saude/shared-types";

const camposConfig = [
  { key: "gestacoes",         label: "Gestações",               min: 0,  max: 20,   step: 1,    placeholder: "Ex: 2",     secao: "demografico" },
  { key: "idade",             label: "Idade (anos)",            min: 1,  max: 120,  step: 1,    placeholder: "Ex: 45",    secao: "demografico" },
  { key: "glicose",           label: "Glicose (mg/dL)",         min: 1,  max: 500,  step: 1,    placeholder: "Ex: 120",   secao: "clinico" },
  { key: "pressao_arterial",  label: "Pressão Arterial (mmHg)", min: 0,  max: 200,  step: 1,    placeholder: "Ex: 70",    secao: "clinico" },
  { key: "espessura_pele",    label: "Espessura da Pele (mm)",  min: 0,  max: 100,  step: 1,    placeholder: "Ex: 28",    secao: "clinico" },
  { key: "insulina",          label: "Insulina (mu U/ml)",      min: 0,  max: 1000, step: 1,    placeholder: "Ex: 0",     secao: "clinico" },
  { key: "imc",               label: "IMC (kg/m²)",             min: 1,  max: 100,  step: 0.1,  placeholder: "Ex: 32.5",  secao: "clinico" },
  { key: "historico_familiar",label: "Histórico Familiar",      min: 0,  max: 2.5,  step: 0.01, placeholder: "Ex: 0.627", secao: "clinico" },
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

function Spinner() {
  return (
    <svg className="animate-spin w-4 h-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  );
}

export default function PreverPage() {
  const [form, setForm] = useState<PacienteDto>(valoresIniciais);
  const { resultado, loading, erro, prever } = usePrever();

  const handleChange = (key: keyof PacienteDto, value: string) => {
    setForm((f) => ({ ...f, [key]: parseFloat(value) || 0 }));
  };

  const demograficos = camposConfig.filter((c) => c.secao === "demografico");
  const clinicos     = camposConfig.filter((c) => c.secao === "clinico");

  const renderCampo = (campo: (typeof camposConfig)[number]) => (
    <div key={campo.key}>
      <label className="block text-sm font-medium text-slate-300 mb-1.5">
        {campo.label}
      </label>
      <input
        type="number"
        min={campo.min}
        max={campo.max}
        step={campo.step}
        value={form[campo.key]}
        placeholder={campo.placeholder}
        onChange={(e) => handleChange(campo.key, e.target.value)}
        className="input-field"
      />
    </div>
  );

  return (
    <div className="space-y-7 max-w-3xl">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">Nova Predição</h1>
        <p className="text-slate-400 text-sm mt-1.5">
          Informe os dados clínicos do paciente para análise de risco de diabetes
        </p>
      </div>

      <div className="card p-6 space-y-6">
        {/* Seção demográfica */}
        <div>
          <div className="flex items-center gap-2.5 mb-4">
            <span className="w-1 h-4 rounded bg-brand-500" />
            <h2 className="text-sm font-semibold text-slate-300">Dados Demográficos</h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {demograficos.map(renderCampo)}
          </div>
        </div>

        <div className="border-t border-slate-700/50" />

        {/* Seção clínica */}
        <div>
          <div className="flex items-center gap-2.5 mb-4">
            <span className="w-1 h-4 rounded bg-teal-500" />
            <h2 className="text-sm font-semibold text-slate-300">Dados Clínicos</h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {clinicos.map(renderCampo)}
          </div>
        </div>

        <button
          onClick={() => prever(form)}
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-semibold
                     bg-gradient-to-r from-brand-700 to-brand-500
                     hover:from-brand-600 hover:to-brand-400 hover:shadow-glow-brand
                     active:scale-[0.98] transition-all duration-200
                     disabled:opacity-50 disabled:cursor-not-allowed
                     disabled:hover:shadow-none disabled:active:scale-100"
        >
          {loading ? <><Spinner /> Analisando...</> : "Analisar Paciente"}
        </button>
      </div>

      {erro && (
        <div className="flex items-start gap-3 bg-red-500/10 border border-red-500/25 rounded-2xl p-4 text-red-400 text-sm animate-slide-up">
          <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {erro}
        </div>
      )}

      {resultado && <ResultadoCard resultado={resultado} />}
    </div>
  );
}
