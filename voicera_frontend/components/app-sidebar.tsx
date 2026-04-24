"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import {
  UserRound,
  Hash,
  ChevronsUpDown,
  LogOut,
  Clock,
  TrendingUp,
  Settings,
  Cpu,
  BookOpen,
  LogIn,
  Layers,
} from "lucide-react"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
} from "@/components/ui/sidebar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { getCurrentUser, clearAuth, type User } from "@/lib/api"

// Menu items for main navigation
const mainNavItems = [
  {
    title: "Agents",
    url: "/assistants",
    icon: UserRound,
  },
  {
    title: "Numbers",
    url: "/numbers",
    icon: Hash,
  },
  {
    title: "Knowledge Base",
    url: "/knowledge-base",
    icon: BookOpen,
  },
  {
    title: "Batches",
    url: "/batches",
    icon: Layers,
  },
  {
    title: "History",
    url: "/history",
    icon: Clock,
  },
  {
    title: "Analytics",
    url: "/analytics",
    icon: TrendingUp,
  },
  {
    title: "Telemetry",
    url: "/telemetry",
    icon: LogIn,
  },
]

// Settings navigation items
const settingsNavItems = [
  {
    title: "Members",
    url: "/members",
    icon: Settings,
  },
  {
    title: "Integrations",
    url: "/integrations",
    icon: Cpu,
  },
]

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const pathname = usePathname()
  const router = useRouter()
  const [user, setUser] = React.useState<User | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)

  // Fetch user data on mount
  React.useEffect(() => {
    async function fetchUser() {
      try {
        const userData = await getCurrentUser()
        setUser(userData)
      } catch (error) {
        console.error("Failed to fetch user:", error)
        // If failed to fetch user, redirect to login
        router.push("/")
      } finally {
        setIsLoading(false)
      }
    }
    fetchUser()
  }, [router])

  const handleLogout = () => {
    clearAuth()
    router.push("/")
  }

  return (
    <Sidebar collapsible="none" className="bg-[#F1EFE8]" {...props}>
      {/* Header - Logo area with proper padding */}
      <SidebarHeader className="bg-[#F1EFE8] px-4 py-4">
        <div className="flex w-full items-center border-b border-[#d6d3cc] pb-2">
          <img
            src="/voicera-logo-black-source.png"
            alt="VoiceRA Logo"
            className="h-20 w-auto max-w-[320px] flex-shrink-0"
            style={{ filter: "brightness(0) saturate(100%)" }}
          />
        </div>
      </SidebarHeader>

      {/* Main Navigation - Primary actions */}
      <SidebarContent className="flex flex-1 flex-col bg-[#F1EFE8] px-3 pt-4">
        <SidebarGroup className="mt-0 pt-0">
          <SidebarGroupLabel className="mb-2 px-2 text-[10px] font-medium uppercase tracking-[0.08em] text-[#8A8882]">
            Navigation
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu className="space-y-1">
              {mainNavItems.map((item) => {
                const isActive = pathname === item.url
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      isActive={isActive}
                      tooltip={item.title}
                      className={
                        isActive
                          ? "w-full rounded-md bg-white font-semibold text-slate-900 hover:bg-white hover:text-slate-900"
                          : "w-full rounded-md text-slate-700 hover:bg-white/70 hover:text-slate-900"
                      }
                    >
                      <Link href={item.url}>
                        <item.icon />
                        <span>{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                )
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <div className="mt-auto pb-2">
          <p className="mb-2 px-2 text-[10px] font-medium uppercase tracking-[0.08em] text-[#8A8882]">
            Settings
          </p>
          <SidebarMenu className="space-y-1">
            {settingsNavItems.map((item) => {
              const isActive = pathname === item.url
              return (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={isActive}
                    tooltip={item.title}
                    className={
                      isActive
                        ? "w-full rounded-md bg-white font-semibold text-slate-900 hover:bg-white hover:text-slate-900"
                        : "w-full rounded-md text-slate-700 hover:bg-white/70 hover:text-slate-900"
                    }
                  >
                    <Link href={item.url}>
                      <item.icon />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              )
            })}
          </SidebarMenu>
        </div>
      </SidebarContent>

      {/* Footer - User profile pinned bottom */}
      <SidebarFooter className="mt-auto bg-[#F1EFE8] px-3 py-3">
        <SidebarSeparator className="my-2 bg-[#DBD7CA]" />

        {/* User Profile */}
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  className="
                    flex items-center w-full gap-3 px-3 py-3 rounded-lg
                    hover:bg-white/70 transition-colors
                    data-[state=open]:bg-white
                  "
                  aria-label="Account menu"
                >
                  <div className="flex h-[30px] w-[30px] shrink-0 items-center justify-center rounded-full border border-[#d0ccc4] bg-[#e8e4dd]">
                    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="#666" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <circle cx="8" cy="5.5" r="2.5" />
                      <path d="M3 13c0-2.76 2.24-5 5-5s5 2.24 5 5" />
                    </svg>
                  </div>
                  <div className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate text-[13px] font-medium text-[#1a1a1a]">
                      {isLoading ? "Loading..." : user?.name || "Unknown"}
                    </span>
                  </div>
                  <ChevronsUpDown className="size-4 shrink-0 text-muted-foreground opacity-35" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                className="w-[--radix-dropdown-menu-trigger-width] min-w-56 rounded-xl shadow-lg"
                side="top"
                align="start"
                sideOffset={8}
              >
                {/* User info header in dropdown */}
                <div className="px-3 py-2 border-b">
                  <p className="text-sm font-medium text-foreground truncate">
                    {user?.name || "Unknown"}
                  </p>
                  <p className="text-xs text-muted-foreground truncate">
                    {user?.email || ""}
                  </p>
                </div>
                <div className="p-1">
                  <DropdownMenuItem
                    className="
                      flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer
                      text-red-600 focus:text-red-600 focus:bg-red-50 
                      transition-colors
                    "
                    onClick={handleLogout}
                  >
                    <LogOut className="size-4" />
                    <span className="text-sm font-medium">Sign out</span>
                  </DropdownMenuItem>
                </div>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  )
}
