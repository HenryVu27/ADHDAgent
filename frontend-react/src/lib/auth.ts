import type { User, OnboardingData } from "@/types"

const STORAGE_KEYS = {
  users: "adhd_agent_users",
  currentUser: "adhd_agent_current_user",
  onboarding: "adhd_agent_onboarding",
  sessions: "adhd_agent_sessions",
} as const

function getUsers(): Record<string, User & { password: string }> {
  const raw = localStorage.getItem(STORAGE_KEYS.users)
  return raw ? JSON.parse(raw) : {}
}

function saveUsers(users: Record<string, User & { password: string }>) {
  localStorage.setItem(STORAGE_KEYS.users, JSON.stringify(users))
}

export function signUp(name: string, email: string, password: string): User {
  const users = getUsers()
  if (users[email]) {
    throw new Error("An account with this email already exists")
  }
  const user: User & { password: string } = {
    name,
    email,
    password,
    createdAt: new Date().toISOString(),
  }
  users[email] = user
  saveUsers(users)
  setCurrentUser({ name, email, createdAt: user.createdAt })
  return { name, email, createdAt: user.createdAt }
}

export function signIn(email: string, password: string): User {
  const users = getUsers()
  const user = users[email]
  if (!user || user.password !== password) {
    throw new Error("Invalid email or password")
  }
  setCurrentUser({ name: user.name, email: user.email, createdAt: user.createdAt })
  return { name: user.name, email: user.email, createdAt: user.createdAt }
}

export function signOut() {
  localStorage.removeItem(STORAGE_KEYS.currentUser)
}

export function getCurrentUser(): User | null {
  const raw = localStorage.getItem(STORAGE_KEYS.currentUser)
  return raw ? JSON.parse(raw) : null
}

function setCurrentUser(user: User) {
  localStorage.setItem(STORAGE_KEYS.currentUser, JSON.stringify(user))
}

export function saveOnboarding(data: OnboardingData) {
  const user = getCurrentUser()
  if (!user) return
  localStorage.setItem(
    `${STORAGE_KEYS.onboarding}_${user.email}`,
    JSON.stringify(data)
  )
}

export function getOnboarding(): OnboardingData | null {
  const user = getCurrentUser()
  if (!user) return null
  const raw = localStorage.getItem(`${STORAGE_KEYS.onboarding}_${user.email}`)
  return raw ? JSON.parse(raw) : null
}

export function hasCompletedOnboarding(): boolean {
  return getOnboarding() !== null
}

export function getSessionStats() {
  const user = getCurrentUser()
  if (!user) return { sessions: 0, strategies: 0, streak: 0 }
  const raw = localStorage.getItem(`${STORAGE_KEYS.sessions}_${user.email}`)
  return raw ? JSON.parse(raw) : { sessions: 0, strategies: 0, streak: 0 }
}

export function updateSessionStats(updates: Partial<{ sessions: number; strategies: number; streak: number }>) {
  const user = getCurrentUser()
  if (!user) return
  const stats = getSessionStats()
  const merged = { ...stats, ...updates }
  localStorage.setItem(
    `${STORAGE_KEYS.sessions}_${user.email}`,
    JSON.stringify(merged)
  )
}

// Active session tracking — persists the current chat session ID per user
export function getActiveSessionId(): string | null {
  const user = getCurrentUser()
  if (!user) return null
  return localStorage.getItem(`${STORAGE_KEYS.sessions}_active_${user.email}`)
}

export function setActiveSessionId(sessionId: string): void {
  const user = getCurrentUser()
  if (!user) return
  localStorage.setItem(`${STORAGE_KEYS.sessions}_active_${user.email}`, sessionId)
}

export function clearActiveSessionId(): void {
  const user = getCurrentUser()
  if (!user) return
  localStorage.removeItem(`${STORAGE_KEYS.sessions}_active_${user.email}`)
}
