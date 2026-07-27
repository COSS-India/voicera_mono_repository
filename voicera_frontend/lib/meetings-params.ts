import type { MeetingsPageParams } from "@/lib/api"

export const MEETINGS_PAGE_SIZE = 50

type ActiveFilter = { field: string; value: string }
type DateRange = { from: Date | undefined; to: Date | undefined }

export function buildMeetingsParams(
  currentPage: number,
  activeFilters: ActiveFilter[],
  dateRange: DateRange,
  dateSortOrder: "latest" | "oldest",
  durationSortOrder: "longest" | "shortest" | null,
  options?: { limit?: number; forExport?: boolean; has_latency_metrics?: boolean; search?: string }
): MeetingsPageParams {
  const params: MeetingsPageParams = {
    page: currentPage,
    limit: options?.limit ?? MEETINGS_PAGE_SIZE,
    forExport: options?.forExport,
    date_sort_order: dateSortOrder,
    duration_sort_order: durationSortOrder,
    has_latency_metrics: options?.has_latency_metrics,
    search: options?.search,
  }

  if (dateRange.from) {
    const from = new Date(dateRange.from)
    from.setHours(0, 0, 0, 0)
    params.date_from = from.toISOString()
  }
  if (dateRange.to) {
    const to = new Date(dateRange.to)
    to.setHours(23, 59, 59, 999)
    params.date_to = to.toISOString()
  }

  for (const filter of activeFilters) {
    switch (filter.field) {
      case "assistant_name":
        params.agent_type = filter.value
        break
      case "call_status":
        params.call_status = filter.value
        break
      case "call_type":
        params.inbound = filter.value.toLowerCase() === "inbound"
        break
      case "from_number":
        params.from_number = filter.value
        break
      case "to_number":
        params.to_number = filter.value
        break
    }
  }

  return params
}
