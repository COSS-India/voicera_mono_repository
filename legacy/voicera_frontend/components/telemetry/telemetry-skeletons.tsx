import { Skeleton } from "@/components/ui/skeleton"

const LIST_ROW_COUNT = 8

export function CallLatencyListSkeleton() {
  return (
    <div className="divide-y divide-slate-100">
      {Array.from({ length: LIST_ROW_COUNT }).map((_, i) => (
        <div
          key={i}
          className="flex items-center gap-4 px-5 py-4"
        >
          <Skeleton className="h-7 w-[84px] shrink-0 rounded-full" />
          <div className="min-w-0 flex-1 basis-0 space-y-1.5">
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-3 w-1/2" />
          </div>
          <Skeleton className="h-3 min-w-0 flex-1 basis-0 max-w-[140px]" />
          <Skeleton className="h-3 w-10 shrink-0" />
          <Skeleton className="h-3 w-6 shrink-0" />
        </div>
      ))}
    </div>
  )
}

export function CallLatencyDetailSkeleton() {
  return (
    <div className="flex-1 overflow-y-auto p-5 lg:p-6 space-y-5 bg-slate-50/50">
      <div className="space-y-2">
        <Skeleton className="h-5 w-48" />
        <Skeleton className="h-3 w-full max-w-md" />
        <Skeleton className="h-3 w-32" />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <div
            key={i}
            className="bg-white rounded-xl border border-slate-200 p-4 flex flex-col min-h-[5.5rem] min-w-0"
          >
            <Skeleton className="h-5 w-24 mb-2" />
            <Skeleton className="h-8 w-20 mt-auto" />
          </div>
        ))}
      </div>
      <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-3">
        <Skeleton className="h-4 w-36" />
        <Skeleton className="h-52 w-full rounded-lg" />
      </div>
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        <Skeleton className="h-11 w-full rounded-none" />
        <div className="p-4 space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-8 w-full" />
          ))}
        </div>
      </div>
    </div>
  )
}

export function GpuTelemetrySkeleton({ cardCount = 2 }: { cardCount?: number }) {
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      {Array.from({ length: cardCount }).map((_, i) => (
        <div
          key={i}
          className="bg-white rounded-xl border border-slate-200 overflow-hidden"
        >
          <div className="p-6 pb-3 space-y-2">
            <Skeleton className="h-5 w-56" />
            <Skeleton className="h-3 w-full max-w-xs" />
          </div>
          <div className="px-6 pb-6 space-y-4">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <Skeleton className="h-44 w-full rounded-lg" />
              <Skeleton className="h-44 w-full rounded-lg" />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Skeleton className="h-3 w-24" />
              <Skeleton className="h-3 w-24" />
            </div>
            <div className="rounded-md border border-slate-200 overflow-hidden">
              <Skeleton className="h-9 w-full rounded-none" />
              <div className="p-3 space-y-2">
                {Array.from({ length: 4 }).map((_, j) => (
                  <Skeleton key={j} className="h-6 w-full" />
                ))}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
