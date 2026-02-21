import {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from "react"
import type { User, OnboardingData } from "@/types"
import * as auth from "@/lib/auth"

interface AuthContextValue {
  user: User | null
  isAuthenticated: boolean
  hasOnboarded: boolean
  signUp: (name: string, email: string, password: string) => void
  signIn: (email: string, password: string) => void
  signOut: () => void
  saveOnboarding: (data: OnboardingData) => void
  getOnboarding: () => OnboardingData | null
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(auth.getCurrentUser)
  const [onboarded, setOnboarded] = useState(auth.hasCompletedOnboarding)

  const handleSignUp = useCallback((name: string, email: string, password: string) => {
    const u = auth.signUp(name, email, password)
    setUser(u)
    setOnboarded(auth.hasCompletedOnboarding())
  }, [])

  const handleSignIn = useCallback((email: string, password: string) => {
    const u = auth.signIn(email, password)
    setUser(u)
    setOnboarded(auth.hasCompletedOnboarding())
  }, [])

  const handleSignOut = useCallback(() => {
    auth.signOut()
    setUser(null)
    setOnboarded(false)
  }, [])

  const handleSaveOnboarding = useCallback((data: OnboardingData) => {
    auth.saveOnboarding(data)
    setOnboarded(true)
  }, [])

  const value: AuthContextValue = {
    user,
    isAuthenticated: !!user,
    hasOnboarded: onboarded,
    signUp: handleSignUp,
    signIn: handleSignIn,
    signOut: handleSignOut,
    saveOnboarding: handleSaveOnboarding,
    getOnboarding: auth.getOnboarding,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used within AuthProvider")
  return ctx
}
