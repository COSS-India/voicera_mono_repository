"use client"

import { Suspense, useEffect, useRef, useState } from "react"
import { usePathname, useSearchParams } from "next/navigation"

const MIN_VISIBLE_MS = 280
const DONE_FADE_MS = 360
const STUCK_RESET_MS = 20000

function isSameRoute(a: URL, b: URL) {
  return (
    a.pathname === b.pathname &&
    a.search === b.search &&
    a.hash === b.hash
  )
}

function NavigationProgressInner() {
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const routeKey = `${pathname}?${searchParams.toString()}`

  const [phase, setPhase] = useState<"idle" | "busy" | "exit">("idle")
  const pendingRef = useRef(false)
  const startTimeRef = useRef(0)
  const prevRouteKeyRef = useRef<string | null>(null)

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      const el = (e.target as Element | null)?.closest?.("a[href]")
      if (!el) return
      const a = el as HTMLAnchorElement
      if (a.target && a.target !== "_self") return
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return

      const raw = a.getAttribute("href")
      if (!raw || raw.startsWith("#")) return
      if (raw.startsWith("mailto:") || raw.startsWith("tel:") || raw.startsWith("sms:")) return

      let dest: URL
      try {
        dest = new URL(raw, window.location.origin)
      } catch {
        return
      }
      if (dest.origin !== window.location.origin) return

      const here = new URL(window.location.href)
      if (isSameRoute(dest, here)) return

      startTimeRef.current = Date.now()
      pendingRef.current = true
      setPhase("busy")
    }

    document.addEventListener("click", onClick, true)
    return () => document.removeEventListener("click", onClick, true)
  }, [])

  useEffect(() => {
    const onPop = () => {
      startTimeRef.current = Date.now()
      pendingRef.current = true
      setPhase("busy")
    }
    window.addEventListener("popstate", onPop)
    return () => window.removeEventListener("popstate", onPop)
  }, [])

  useEffect(() => {
    if (prevRouteKeyRef.current === null) {
      prevRouteKeyRef.current = routeKey
      return
    }
    if (prevRouteKeyRef.current === routeKey) return

    const hadPending = pendingRef.current
    prevRouteKeyRef.current = routeKey

    if (!hadPending) return

    const elapsed = Date.now() - startTimeRef.current
    const wait = Math.max(0, MIN_VISIBLE_MS - elapsed)
    const t = window.setTimeout(() => {
      pendingRef.current = false
      setPhase("exit")
      window.setTimeout(() => setPhase("idle"), DONE_FADE_MS)
    }, wait)
    return () => window.clearTimeout(t)
  }, [routeKey])

  useEffect(() => {
    if (phase !== "busy") return
    const t = window.setTimeout(() => {
      if (pendingRef.current) {
        pendingRef.current = false
        setPhase("idle")
      }
    }, STUCK_RESET_MS)
    return () => window.clearTimeout(t)
  }, [phase])

  if (phase === "idle") return null

  return (
    <div
      className="pointer-events-none fixed inset-x-0 top-0 z-[99999] h-[3px] overflow-hidden"
      aria-hidden
    >
      {phase === "busy" ? (
        <div className="relative h-full w-full bg-indigo-500/15">
          <div className="nav-progress-indeterminate h-full w-[42%] rounded-r-sm bg-indigo-600 shadow-[0_0_14px_rgba(79,70,229,0.55)]" />
        </div>
      ) : (
        <div className="nav-progress-exit h-full w-full origin-left bg-indigo-600 shadow-[0_0_14px_rgba(79,70,229,0.45)]" />
      )}
    </div>
  )
}

export function NavigationProgress() {
  return (
    <Suspense fallback={null}>
      <NavigationProgressInner />
    </Suspense>
  )
}
