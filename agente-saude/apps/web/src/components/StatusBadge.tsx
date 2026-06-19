export default function StatusBadge({ disponivel }: { disponivel: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold border ${
        disponivel
          ? "bg-green-500/10 text-green-400 border-green-500/30"
          : "bg-red-500/10  text-red-400   border-red-500/30"
      }`}
    >
      <span className="relative flex h-2 w-2">
        {disponivel && (
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-60" />
        )}
        <span
          className={`relative inline-flex h-2 w-2 rounded-full ${
            disponivel ? "bg-green-400" : "bg-red-400"
          }`}
        />
      </span>
      {disponivel ? "Online" : "Offline"}
    </span>
  );
}
