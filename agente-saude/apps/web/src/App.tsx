import { Routes, Route, NavLink } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
import PreverPage from "./pages/PreverPage";
import HistoricoPage from "./pages/HistoricoPage";

function HeartIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 21.593c-5.63-5.539-11-10.297-11-14.402 0-3.791 3.068-5.191 5.281-5.191 1.312 0 4.151.501 5.719 4.457 1.59-3.968 4.464-4.447 5.726-4.447 2.54 0 5.274 1.621 5.274 5.181 0 4.069-5.136 8.625-11 14.402z" />
    </svg>
  );
}

export default function App() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
      isActive
        ? "bg-brand-600/20 text-white border border-brand-500/30 shadow-glow-brand"
        : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/80"
    }`;

  return (
    <div className="min-h-screen bg-[#0b1120] text-white">
      <nav className="sticky top-0 z-50 bg-slate-900/80 backdrop-blur-md border-b border-slate-800/80">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center gap-2">
          <div className="flex items-center gap-2.5 mr-6">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center shadow-glow-brand">
              <HeartIcon />
            </div>
            <span className="font-bold text-white text-base tracking-tight">
              Agente <span className="text-brand-400">Saúde</span>
            </span>
          </div>

          <div className="flex items-center gap-1">
            <NavLink to="/" end className={linkClass}>
              Dashboard
            </NavLink>
            <NavLink to="/prever" className={linkClass}>
              Nova Predição
            </NavLink>
            <NavLink to="/historico" className={linkClass}>
              Histórico
            </NavLink>
          </div>
        </div>
      </nav>

      <main className="max-w-6xl mx-auto px-6 py-8 animate-fade-in">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/prever" element={<PreverPage />} />
          <Route path="/historico" element={<HistoricoPage />} />
        </Routes>
      </main>
    </div>
  );
}
