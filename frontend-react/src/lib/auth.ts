import type { User, OnboardingData, TokenResponse } from "@/types"

const TOKEN_KEY = "adhd_token"
const USER_KEY = "adhd_user"

// ---------------------------------------------------------------------------
// Token storage
// ---------------------------------------------------------------------------

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function authPost(path: string, body: object): Promise<TokenResponse> {
  const res = await fetch(`/api/auth${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `Request failed: ${res.status}`)
  }
  return res.json()
}

export async function getMe(): Promise<User | null> {
  const token = getToken()
  if (!token) return null
  const res = await fetch("/api/auth/me", {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!res.ok) {
    clearToken()
    return null
  }
  const data = await res.json()
  return { id: data.id, email: data.email, name: data.name ?? null, createdAt: data.created_at }
}

// ---------------------------------------------------------------------------
// Auth actions
// ---------------------------------------------------------------------------

export async function signUp(email: string, password: string, name: string): Promise<User> {
  const { access_token } = await authPost("/register", { email, password, name })
  setToken(access_token)
  const user = await getMe()
  if (!user) throw new Error("Registration succeeded but could not fetch user")
  localStorage.setItem(USER_KEY, JSON.stringify(user))
  return user
}

export async function signIn(email: string, password: string): Promise<User> {
  const { access_token } = await authPost("/login", { email, password })
  setToken(access_token)
  const user = await getMe()
  if (!user) throw new Error("Login succeeded but could not fetch user")
  localStorage.setItem(USER_KEY, JSON.stringify(user))
  return user
}

export function signOut(): void {
  clearToken()
  localStorage.removeItem(USER_KEY)
}

export function getCachedUser(): User | null {
  const raw = localStorage.getItem(USER_KEY)
  return raw ? JSON.parse(raw) : null
}

// ---------------------------------------------------------------------------
// Onboarding (still localStorage — no backend model for it yet)
// ---------------------------------------------------------------------------

export function saveOnboarding(data: OnboardingData): void {
  const user = getCachedUser()
  if (!user) return
  localStorage.setItem(`adhd_onboarding_${user.id}`, JSON.stringify(data))
}

export function getOnboarding(): OnboardingData | null {
  const user = getCachedUser()
  if (!user) return null
  const raw = localStorage.getItem(`adhd_onboarding_${user.id}`)
  return raw ? JSON.parse(raw) : null
}

export function hasCompletedOnboarding(): boolean {
  return getOnboarding() !== null
}

// ---------------------------------------------------------------------------
// Session stats (keyed by user.id — migrated from email-keyed storage)
// ---------------------------------------------------------------------------

export function getSessionStats() {
  const user = getCachedUser()
  if (!user) return { sessions: 0, strategies: 0, streak: 0 }
  const raw = localStorage.getItem(`adhd_sessions_stats_${user.id}`)
  return raw ? JSON.parse(raw) : { sessions: 0, strategies: 0, streak: 0 }
}

export function updateSessionStats(updates: Partial<{ sessions: number; strategies: number; streak: number }>) {
  const user = getCachedUser()
  if (!user) return
  const stats = getSessionStats()
  localStorage.setItem(`adhd_sessions_stats_${user.id}`, JSON.stringify({ ...stats, ...updates }))
}

// ---------------------------------------------------------------------------
// Active session tracking (keyed by user.id)
// ---------------------------------------------------------------------------

export function getActiveSessionId(): string | null {
  const user = getCachedUser()
  if (!user) return null
  return localStorage.getItem(`adhd_sessions_active_${user.id}`)
}

export function setActiveSessionId(sessionId: string): void {
  const user = getCachedUser()
  if (!user) return
  localStorage.setItem(`adhd_sessions_active_${user.id}`, sessionId)
}

export function clearActiveSessionId(): void {
  const user = getCachedUser()
  if (!user) return
  localStorage.removeItem(`adhd_sessions_active_${user.id}`)
}
