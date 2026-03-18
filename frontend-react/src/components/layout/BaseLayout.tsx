import { Link, useLocation, useNavigate } from "react-router-dom"
import { Sprout, LayoutDashboard, MessageCircle, BookOpen, User, LogOut } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useAuth } from "@/hooks/use-auth"
import type { ReactNode } from "react"

const navItems = [
  { to: "/dashboard", label: "Dashboard", mobileLabel: "Home", icon: LayoutDashboard },
  { to: "/chat", label: "Chat with Ally", mobileLabel: "Ally", icon: MessageCircle },
  { to: "/resources", label: "Resources", mobileLabel: "Resources", icon: BookOpen },
  { to: "/profile", label: "Profile", mobileLabel: "Profile", icon: User },
]

export function BaseLayout({ children }: { children: ReactNode }) {
  const { user, signOut } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()

  const handleSignOut = () => {
    signOut()
    navigate("/")
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4">
          <Link to="/dashboard" className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-coach">
              <Sprout className="h-5 w-5 text-coach-foreground" />
            </div>
          </Link>

          <nav className="hidden items-center gap-1 md:flex">
            {navItems.map(({ to, label, icon: Icon }) => {
              const isActive = location.pathname === to
              return (
                <Link key={to} to={to}>
                  <Button
                    variant={isActive ? "secondary" : "ghost"}
                    size="sm"
                    className="gap-2"
                  >
                    <Icon className="h-4 w-4" />
                    {label}
                  </Button>
                </Link>
              )
            })}
          </nav>

          <div className="flex items-center gap-3">
            {user && (
              <span className="hidden text-sm text-muted-foreground md:block">
                {user.name}
              </span>
            )}
            <Button variant="ghost" size="sm" onClick={handleSignOut} className="gap-2">
              <LogOut className="h-4 w-4" />
              <span className="hidden md:inline">Sign Out</span>
            </Button>
          </div>
        </div>

        {/* Mobile nav */}
        <nav className="flex border-t border-border/50 md:hidden">
          {navItems.map(({ to, mobileLabel, icon: Icon }) => {
            const isActive = location.pathname === to
            return (
              <Link
                key={to}
                to={to}
                className={`flex flex-1 flex-col items-center gap-0.5 py-2 text-xs transition-colors ${
                  isActive
                    ? "text-primary"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <Icon className="h-4 w-4" />
                {mobileLabel}
              </Link>
            )
          })}
        </nav>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6">{children}</main>
    </div>
  )
}
