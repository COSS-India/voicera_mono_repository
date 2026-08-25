import { useQuery, keepPreviousData } from "@tanstack/react-query"
import {
  getMeetingsPage,
  getMeetingFilterOptions,
  type MeetingsPageParams,
} from "@/lib/api"

export const meetingsQueryKey = (params: MeetingsPageParams) =>
  ["meetings", params] as const

export const meetingFilterOptionsQueryKey = () =>
  ["meetings", "filter-options"] as const

export function useMeetingsQuery(params: MeetingsPageParams) {
  return useQuery({
    queryKey: meetingsQueryKey(params),
    queryFn: () => getMeetingsPage(params),
    placeholderData: keepPreviousData,
    staleTime: 5 * 60 * 1000,
  })
}

export function useMeetingFilterOptionsQuery() {
  return useQuery({
    queryKey: meetingFilterOptionsQueryKey(),
    queryFn: getMeetingFilterOptions,
    staleTime: 10 * 60 * 1000,
  })
}
