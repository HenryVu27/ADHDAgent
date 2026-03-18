import { Routes, Route, Navigate } from "react-router-dom"
import { AnimatePresence } from "framer-motion"
import { useAuth } from "@/hooks/use-auth"
import { BaseLayout } from "@/components/layout/BaseLayout"
import { AuthLayout } from "@/components/layout/AuthLayout"
import { PageTransition } from "@/components/layout/PageTransition"
import { LandingPage } from "@/pages/LandingPage"
import { SignInPage } from "@/pages/SignInPage"
import { SignUpPage } from "@/pages/SignUpPage"
import { OnboardingPage } from "@/pages/OnboardingPage"
import { DashboardPage } from "@/pages/DashboardPage"
import { ChatPage } from "@/pages/ChatPage"
import { ResourcesPage } from "@/pages/ResourcesPage"
import { ProfilePage } from "@/pages/ProfilePage"
import { ObservabilityPage } from "@/pages/ObservabilityPage"
import type { ReactNode } from "react"

function ProtectedRoute({ children }: { children: ReactNode }) {
  const { isAuthenticated, hasOnboarded } = useAuth()
  if (!isAuthenticated) return <Navigate to="/" replace />
  if (!hasOnboarded) return <Navigate to="/onboarding" replace />
  return <>{children}</>
}

function RequireAuth({ children }: { children: ReactNode }) {
  const { isAuthenticated } = useAuth()
  if (!isAuthenticated) return <Navigate to="/" replace />
  return <>{children}</>
}

function AuthRoute({ children }: { children: ReactNode }) {
  const { isAuthenticated, hasOnboarded } = useAuth()
  if (isAuthenticated && hasOnboarded) return <Navigate to="/dashboard" replace />
  if (isAuthenticated && !hasOnboarded) return <Navigate to="/onboarding" replace />
  return <>{children}</>
}

export function App() {
  return (
    <AnimatePresence mode="wait">
      <Routes>
        {/* Public */}
        <Route
          path="/"
          element={
            <AuthRoute>
              <PageTransition>
                <LandingPage />
              </PageTransition>
            </AuthRoute>
          }
        />
        <Route
          path="/signin"
          element={
            <AuthRoute>
              <AuthLayout>
                <PageTransition>
                  <SignInPage />
                </PageTransition>
              </AuthLayout>
            </AuthRoute>
          }
        />
        <Route
          path="/signup"
          element={
            <AuthRoute>
              <AuthLayout>
                <PageTransition>
                  <SignUpPage />
                </PageTransition>
              </AuthLayout>
            </AuthRoute>
          }
        />

        {/* Onboarding (authed but not onboarded) */}
        <Route
          path="/onboarding"
          element={
            <PageTransition>
              <OnboardingPage />
            </PageTransition>
          }
        />

        {/* Protected */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <BaseLayout>
                <PageTransition>
                  <DashboardPage />
                </PageTransition>
              </BaseLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/chat"
          element={
            <ProtectedRoute>
              <BaseLayout>
                <PageTransition>
                  <ChatPage />
                </PageTransition>
              </BaseLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/resources"
          element={
            <ProtectedRoute>
              <BaseLayout>
                <PageTransition>
                  <ResourcesPage />
                </PageTransition>
              </BaseLayout>
            </ProtectedRoute>
          }
        />
        <Route
          path="/profile"
          element={
            <ProtectedRoute>
              <BaseLayout>
                <PageTransition>
                  <ProfilePage />
                </PageTransition>
              </BaseLayout>
            </ProtectedRoute>
          }
        />

        {/* Admin (auth required, no onboarding required) */}
        <Route
          path="/observability"
          element={
            <RequireAuth>
              <ObservabilityPage />
            </RequireAuth>
          }
        />

        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AnimatePresence>
  )
}
