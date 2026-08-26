import { useQuery } from '@tanstack/react-query'
import axios from '../lib/axios'

export interface BackendHealth {
  /** True only while GET /health is actually answering. */
  isConnected: boolean
  /** Timestamp of the last successful probe, or null if there has never been one. */
  lastCheckedAt: number | null
}

/**
 * Real backend connection status, derived from `GET /health`
 * (unified_server.jl:342).
 *
 * This replaces a `useWebSocket` stub that hardcoded `isConnected = true`
 * "to not block UI" with its `connect()` call commented out. There is no
 * `@websocket` route in the backend at all, so the flag could never become
 * false: the dashboard reported "Connected" while every data panel failed.
 *
 * Freshness elsewhere in the app comes from react-query polling, not a socket.
 */
export function useBackendHealth(intervalMs = 5000): BackendHealth {
  const { isSuccess, dataUpdatedAt } = useQuery({
    queryKey: ['backend-health'],
    queryFn: async () => (await axios.get('/health')).data,
    refetchInterval: intervalMs,
    // A failed probe is the signal, not an error to retry away.
    retry: false,
  })

  return {
    isConnected: isSuccess,
    lastCheckedAt: dataUpdatedAt || null,
  }
}
