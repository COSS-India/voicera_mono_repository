"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import {
  Bot,
  Hash,
  ChevronsUpDown,
  LogOut,
  History,
  BarChart3,
  Contact,
  Plug,
  BookOpen,
  Activity,
  Layers3,
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
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { getCurrentUser, clearAuth, type User } from "@/lib/api"

// Menu items for main navigation
const mainNavItems = [
  {
    title: "Agents",
    url: "/assistants",
    icon: Bot,
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
    icon: Layers3,
  },
  {
    title: "History",
    url: "/history",
    icon: History,
  },
  {
    title: "Analytics",
    url: "/analytics",
    icon: BarChart3,
  },
  {
    title: "Telemetry",
    url: "/telemetry",
    icon: Activity,
  },
]

// Settings navigation items
const settingsNavItems = [
  {
    title: "Members",
    url: "/members",
    icon: Contact,
  },
  {
    title: "Integrations",
    url: "/integrations",
    icon: Plug,
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

  // Get initials from name
  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2)
  }

  return (
    <Sidebar collapsible="none" {...props}>
      {/* Header - Logo area with proper padding */}
      <SidebarHeader className="px-4 py-4">
        <div className="flex items-center">
          <img
            src="/ekstep.svg"
            alt="Framewise Logo"
            className="h-8 w-auto max-w-full flex-shrink-0"
          />
        </div>
      </SidebarHeader>

      {/* Main Navigation - Primary actions */}
      <SidebarContent className="px-3">
        <SidebarGroup>
          <SidebarGroupLabel className="text-xs font-medium text-muted-foreground uppercase tracking-wider px-2 mb-2">
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
                          ? "bg-primary text-primary-foreground shadow-sm hover:bg-primary hover:text-primary-foreground"
                          : ""
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
      </SidebarContent>

      {/* Footer - Settings + User profile */}
      <SidebarFooter className="px-3 py-3 mt-auto">
        {/* Settings Navigation */}
        <div className="mb-3">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider px-2 mb-2">
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
                        ? "bg-primary text-primary-foreground shadow-sm hover:bg-primary hover:text-primary-foreground"
                        : ""
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

        <SidebarSeparator className="my-2" />

        {/* User Profile */}
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  className="
                    flex items-center w-full gap-3 px-3 py-3 rounded-lg
                    hover:bg-accent transition-colors
                    data-[state=open]:bg-accent
                  "
                  aria-label="Account menu"
                >
                  <Avatar className="h-10 w-10 rounded-full shrink-0 ring-2 ring-background shadow-sm">
                    <AvatarFallback className="rounded-full bg-gradient-to-br from-blue-500 to-blue-600 text-white text-sm font-semibold">
                      {isLoading ? "..." : user ? getInitials(user.name) : "?"}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex flex-col min-w-0 flex-1">
                    <span className="truncate font-semibold text-sm text-foreground">
                      {isLoading ? "Loading..." : user?.name || "Unknown"}
                    </span>
                    <span className="truncate text-xs text-muted-foreground">
                      {isLoading ? "" : user?.email || ""}
                    </span>
                  </div>
                  <ChevronsUpDown className="size-4 text-muted-foreground shrink-0" />
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
