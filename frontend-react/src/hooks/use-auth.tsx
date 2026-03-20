import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react"
import type { User, OnboardingData } from "@/types"
import * as auth from "@/lib/auth"

interface AuthContextValue {
  user: User | null
  isAuthenticated: boolean
  hasOnboarded: boolean
  isLoading: boolean
  error: string | null
  signUp: (email: string, password: string, name: string) => Promise<void>
  signIn: (email: string, password: string) => Promise<void>
  signOut: () => void
  saveOnboarding: (data: OnboardingData) => void
  getOnboarding: () => OnboardingData | null
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(auth.getCachedUser)
  const [onboarded, setOnboarded] = useState(auth.hasCompletedOnboarding)
  const [isLoading, setIsLoading] = useState(!!auth.getToken())
  const [error, setError] = useState<string | null>(null)

  // Restore user from token on mount
  useEffect(() => {
    if (!auth.getToken()) {
      setIsLoading(false)
      return
    }
    auth.getMe().then((u) => {
      setUser(u)
      setOnboarded(auth.hasCompletedOnboarding())
    }).finally(() => setIsLoading(false))
  }, [])

  const handleSignUp = useCallback(async (email: string, password: string, name: string) => {
    setError(null)
    setIsLoading(true)
    try {
      const u = await auth.signUp(email, password, name)
      setUser(u)
      setOnboarded(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Sign up failed")
      throw e
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleSignIn = useCallback(async (email: string, password: string) => {
    setError(null)
    setIsLoading(true)
    try {
      const u = await auth.signIn(email, password)
      setUser(u)
      setOnboarded(auth.hasCompletedOnboarding())
    } catch (e) {
      setError(e instanceof Error ? e.message : "Sign in failed")
      throw e
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleSignOut = useCallback(() => {
    auth.signOut()
    setUser(null)
    setOnboarded(false)
    setError(null)
  }, [])

  const handleSaveOnboarding = useCallback((data: OnboardingData) => {
    auth.saveOnboarding(data)
    setOnboarded(true)
  }, [])

  const value: AuthContextValue = {
    user,
    isAuthenticated: !!user,
    hasOnboarded: onboarded,
    isLoading,
    error,
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
