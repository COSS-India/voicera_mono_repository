import { useQuery } from "@tanstack/react-query"
import { getAgents } from "@/lib/api"

export const agentsQueryKey = (orgId: string) => ["agents", orgId] as const

export function useAgentsQuery(orgId: string | undefined) {
  return useQuery({
    queryKey: agentsQueryKey(orgId ?? ""),
    queryFn: () => getAgents(orgId!),
    enabled: !!orgId,
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
    refetchOnMount: false,
    refetchOnWindowFocus: false,
  })
}
