export default function DashboardLoading() {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-6 p-6 md:p-8">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-2">
          <div className="route-skeleton h-8 w-40" />
          <div className="route-skeleton h-4 w-72 max-w-full" />
        </div>
        <div className="route-skeleton h-10 w-32 rounded-lg" />
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            className="route-skeleton h-36 rounded-xl border border-border/40 bg-card/30"
          />
        ))}
      </div>

      <div className="route-skeleton h-px w-full max-w-2xl opacity-60" />

      <div className="space-y-3">
        <div className="route-skeleton h-5 w-48" />
        <div className="route-skeleton h-24 w-full max-w-3xl rounded-xl" />
      </div>
    </div>
  )
}
