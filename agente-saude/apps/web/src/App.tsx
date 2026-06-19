import { Routes, Route, NavLink } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
import PreverPage from "./pages/PreverPage";
import HistoricoPage from "./pages/HistoricoPage";

export default function App() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
      isActive
        ? "bg-brand-600 text-white"
        : "text-slate-300 hover:bg-slate-700 hover:text-white"
    }`;

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Navbar */}
      <nav className="bg-slate-800 border-b border-slate-700 px-6 py-3 flex items-center gap-6">
        <span className="text-brand-500 font-bold text-lg mr-4">
          🏥 Agente Saúde
        </span>
        <NavLink to="/" className={linkClass}>
          Dashboard
        </NavLink>
        <NavLink to="/prever" className={linkClass}>
          Nova Predição
        </NavLink>
        <NavLink to="/historico" className={linkClass}>
          Histórico
        </NavLink>
      </nav>

      {/* Conteúdo */}
      <main className="max-w-6xl mx-auto px-6 py-8">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/prever" element={<PreverPage />} />
          <Route path="/historico" element={<HistoricoPage />} />
        </Routes>
      </main>
    </div>
  );
}
