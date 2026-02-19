import { motion, AnimatePresence } from "framer-motion"
import { X, Code2, Clock, Shield, BookOpen, Cpu } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import type { PipelineTrace } from "@/types"

const stepNames: Record<string, string> = {
  input_rails: "Input Rails",
  phase_manager: "Phase Manager",
  rag_retrieval: "RAG",
  response_generation: "Response",
  output_rails: "Output Rails",
}

interface Props {
  trace: PipelineTrace | null
  isOpen: boolean
  onClose: () => void
}

export function DevPanel({ trace, isOpen, onClose }: Props) {
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ x: "100%" }}
          animate={{ x: 0 }}
          exit={{ x: "100%" }}
          transition={{ type: "spring", damping: 25, stiffness: 200 }}
          className="fixed inset-y-0 right-0 z-50 w-full max-w-md border-l border-border bg-card shadow-xl overflow-y-auto md:top-16"
        >
          <div className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-card p-4">
            <div className="flex items-center gap-2">
              <Code2 className="h-4 w-4 text-primary" />
              <span className="font-semibold tracking-tight">Pipeline Trace</span>
            </div>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>

          {!trace ? (
            <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
              Send a message to see the pipeline trace.
            </div>
          ) : (
            <div className="space-y-4 p-4">
              {/* Timing */}
              <section>
                <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                  <Clock className="h-4 w-4" />
                  Timing
                </div>
                <div className="space-y-1">
                  {trace.steps.map((step) => (
                    <div key={step.name} className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">
                        {stepNames[step.name] || step.name}
                      </span>
                      <span className="font-mono text-xs">{step.duration_ms.toFixed(0)}ms</span>
                    </div>
                  ))}
                  <Separator className="my-1" />
                  <div className="flex items-center justify-between text-sm font-medium">
                    <span>Total</span>
                    <span className="font-mono text-xs">
                      {trace.total_duration_ms.toFixed(0)}ms
                    </span>
                  </div>
                </div>
              </section>

              <Separator />

              {/* Input Rails */}
              {trace.input_check && (
                <section>
                  <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                    <Shield className="h-4 w-4" />
                    Input Rails
                  </div>
                  <Badge
                    variant={trace.input_check.is_allowed ? "secondary" : "destructive"}
                  >
                    {trace.input_check.is_allowed ? "allowed" : trace.input_check.blocked_reason}
                  </Badge>
                </section>
              )}

              <Separator />

              {/* Phase Decision */}
              {trace.phase_decision && (
                <section>
                  <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                    <Cpu className="h-4 w-4" />
                    Phase Decision
                  </div>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Phase</span>
                      <Badge variant="outline">{trace.phase_decision.phase}</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Agent</span>
                      <span>{trace.phase_decision.agent}</span>
                    </div>
                    {trace.phase_decision.phase_changed && (
                      <Badge className="bg-warm text-warm-foreground">Phase Changed</Badge>
                    )}
                    {trace.phase_decision.directives.length > 0 && (
                      <div className="mt-1">
                        <span className="text-muted-foreground">Directives:</span>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {trace.phase_decision.directives.map((d, i) => (
                            <Badge key={i} variant="outline" className="text-xs">
                              {d}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </section>
              )}

              <Separator />

              {/* RAG Results */}
              <section>
                <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                  <BookOpen className="h-4 w-4" />
                  RAG Results ({trace.retrieval_results.length})
                </div>
                {trace.retrieval_results.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No documents retrieved</p>
                ) : (
                  <div className="space-y-2">
                    {trace.retrieval_results.map((r) => (
                      <div
                        key={r.document_id}
                        className="rounded-md border border-border bg-muted/50 p-2"
                      >
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-medium">{r.document_name}</span>
                          <span className="font-mono text-muted-foreground">
                            {r.score.toFixed(3)}
                          </span>
                        </div>
                        <div className="mt-1 flex flex-wrap gap-1">
                          {r.tags.map((tag) => (
                            <Badge key={tag} variant="secondary" className="text-[10px] px-1.5 py-0">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  )
}
