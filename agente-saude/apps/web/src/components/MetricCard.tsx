type Props = {
  label: string;
  value: number;
  color: "blue" | "purple" | "green" | "amber";
};

const colorMap = {
  blue: "text-blue-400   bg-blue-900/30   border-blue-800",
  purple: "text-purple-400 bg-purple-900/30 border-purple-800",
  green: "text-green-400  bg-green-900/30  border-green-800",
  amber: "text-amber-400  bg-amber-900/30  border-amber-800",
};

export default function MetricCard({ label, value, color }: Props) {
  return (
    <div className={`rounded-xl p-4 border ${colorMap[color]}`}>
      <p className="text-xs text-slate-400 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${colorMap[color].split(" ")[0]}`}>
        {(value * 100).toFixed(1)}%
      </p>
    </div>
  );
}
