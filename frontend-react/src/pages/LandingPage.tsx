import { Link } from "react-router-dom"
import { motion } from "framer-motion"
import { Sprout, Shield, BookOpen, MessageCircle, Sparkles, ArrowRight, Brain } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"

const features = [
  {
    icon: MessageCircle,
    title: "Guided Coaching",
    description:
      "Evidence-based strategies delivered through warm, supportive conversations tailored to your family.",
  },
  {
    icon: Shield,
    title: "Clinician-Safe Bounds",
    description:
      "Every response stays within clinician-informed guardrails. No medication advice, no diagnosis.",
  },
  {
    icon: Brain,
    title: "AI-Powered Understanding",
    description:
      "Advanced NLP extracts context from your messages and matches you with the right strategies.",
  },
  {
    icon: BookOpen,
    title: "Vetted Knowledge Base",
    description:
      "All strategies sourced from peer-reviewed research and clinical best practices for ADHD.",
  },
]

const container = {
  hidden: {},
  show: {
    transition: { staggerChildren: 0.1 },
  },
}

const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
}

export function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Hero */}
      <section className="relative overflow-hidden px-4 pb-16 pt-20 md:pb-24 md:pt-32">
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_top,oklch(0.92_0.04_155)_0%,transparent_60%)]" />
        <div className="mx-auto max-w-4xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-coach shadow-lg md:h-20 md:w-20">
              <Sprout className="h-8 w-8 text-coach-foreground md:h-10 md:w-10" />
            </div>
            <h1 className="mb-4 text-4xl font-bold tracking-tight md:text-6xl">
              Your ADHD
              <span className="text-primary"> Parenting Coach</span>
            </h1>
            <p className="mx-auto mb-8 max-w-2xl text-lg text-muted-foreground leading-relaxed md:text-xl">
              Evidence-based support for parents of children with ADHD.
              Personalized strategies, guided conversations, and a knowledge base
              you can trust — all within clinician-informed bounds.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="flex flex-col items-center gap-3 sm:flex-row sm:justify-center"
          >
            <Link to="/signup">
              <Button size="lg" className="gap-2 px-8 text-base shadow-md">
                Get Started Free
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
            <Link to="/signin">
              <Button variant="outline" size="lg" className="px-8 text-base">
                Sign In
              </Button>
            </Link>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="mt-12 flex items-center justify-center gap-2 text-sm text-muted-foreground"
          >
            <Sparkles className="h-4 w-4 text-warm" />
            <span>Powered by responsible AI with clinician-defined guardrails</span>
          </motion.div>
        </div>
      </section>

      {/* Features */}
      <section className="px-4 pb-20">
        <div className="mx-auto max-w-5xl">
          <motion.div
            variants={container}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
            className="grid gap-6 sm:grid-cols-2"
          >
            {features.map((feature) => (
              <motion.div key={feature.title} variants={item}>
                <Card className="h-full transition-shadow duration-200 hover:shadow-md">
                  <CardContent className="flex gap-4 p-6">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10">
                      <feature.icon className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <h3 className="mb-1 font-semibold tracking-tight">
                        {feature.title}
                      </h3>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        {feature.description}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* How It Works */}
      <section className="border-t border-border/50 bg-muted/30 px-4 py-20">
        <div className="mx-auto max-w-4xl text-center">
          <h2 className="mb-12 text-2xl font-semibold tracking-tight md:text-3xl">
            How It Works
          </h2>
          <div className="grid gap-8 md:grid-cols-3">
            {[
              {
                step: "1",
                title: "Share Your Story",
                desc: "Tell us about your child and the challenges your family faces.",
              },
              {
                step: "2",
                title: "Get Matched",
                desc: "Our AI matches your situation with evidence-based strategies.",
              },
              {
                step: "3",
                title: "Grow Together",
                desc: "Track progress, adjust strategies, and celebrate wins.",
              },
            ].map((s) => (
              <motion.div
                key={s.step}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: Number(s.step) * 0.1 }}
                className="flex flex-col items-center"
              >
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-coach text-lg font-bold text-coach-foreground">
                  {s.step}
                </div>
                <h3 className="mb-2 font-semibold">{s.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {s.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/50 px-4 py-8">
        <div className="mx-auto flex max-w-4xl items-center justify-between text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <Sprout className="h-4 w-4 text-coach" />
            <span>FirstThen</span>
          </div>
          <span>A FirstThen prototype</span>
        </div>
      </footer>
    </div>
  )
}
