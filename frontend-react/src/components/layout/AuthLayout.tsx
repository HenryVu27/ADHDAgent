import { Brain } from "lucide-react"
import { Link } from "react-router-dom"
import type { ReactNode } from "react"

export function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background px-4">
      <Link to="/" className="mb-8 flex items-center gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary shadow-md">
          <Brain className="h-6 w-6 text-primary-foreground" />
        </div>
        <span className="text-2xl font-semibold tracking-tight">ADHDAgent</span>
      </Link>
      <div className="w-full max-w-md">{children}</div>
    </div>
  )
}
