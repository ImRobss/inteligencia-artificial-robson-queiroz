export default function StatusBadge({ disponivel }: { disponivel: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium
      ${
        disponivel
          ? "bg-green-900/40 text-green-400 border border-green-700"
          : "bg-red-900/40   text-red-400   border border-red-700"
      }`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${disponivel ? "bg-green-400" : "bg-red-400"}`}
      />
      {disponivel ? "Online" : "Offline"}
    </span>
  );
}
