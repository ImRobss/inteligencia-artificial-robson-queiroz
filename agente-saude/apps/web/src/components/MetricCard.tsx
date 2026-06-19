type Props = {
  label: string;
  value: number;
  color: "blue" | "purple" | "green" | "amber";
};

const colorMap = {
  blue: {
    text:   "text-blue-400",
    border: "border-blue-500/25",
    bg:     "bg-blue-500/5",
    bar:    "bg-gradient-to-r from-blue-700 to-blue-400",
    glow:   "hover:shadow-[0_0_20px_rgba(59,130,246,0.12)]",
  },
  purple: {
    text:   "text-purple-400",
    border: "border-purple-500/25",
    bg:     "bg-purple-500/5",
    bar:    "bg-gradient-to-r from-purple-700 to-purple-400",
    glow:   "hover:shadow-[0_0_20px_rgba(168,85,247,0.12)]",
  },
  green: {
    text:   "text-green-400",
    border: "border-green-500/25",
    bg:     "bg-green-500/5",
    bar:    "bg-gradient-to-r from-green-700 to-green-400",
    glow:   "hover:shadow-[0_0_20px_rgba(34,197,94,0.12)]",
  },
  amber: {
    text:   "text-amber-400",
    border: "border-amber-500/25",
    bg:     "bg-amber-500/5",
    bar:    "bg-gradient-to-r from-amber-700 to-amber-400",
    glow:   "hover:shadow-[0_0_20px_rgba(245,158,11,0.12)]",
  },
};

export default function MetricCard({ label, value, color }: Props) {
  const c = colorMap[color];
  const pct = (value * 100).toFixed(1);

  return (
    <div
      className={`rounded-2xl p-5 border ${c.border} ${c.bg} ${c.glow}
                  hover:scale-[1.02] transition-all duration-200`}
    >
      <p className="section-label mb-3">{label}</p>
      <p className={`text-3xl font-bold tracking-tight ${c.text} mb-4`}>{pct}%</p>
      <div className="w-full bg-slate-700/40 rounded-full h-1.5 overflow-hidden">
        <div
          className={`h-1.5 rounded-full ${c.bar} transition-all duration-700`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
