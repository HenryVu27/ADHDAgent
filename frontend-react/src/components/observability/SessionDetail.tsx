import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type {
  SessionDetailResponse,
  EnrichedTrace,
  TurnAnalysis,
  ObservabilityEvent,
  AgentReasoningStep,
} from "@/types"

interface SessionDetailProps {
  data: SessionDetailResponse
  onBack: () => void
  onAnalyze: () => void
  analyzing: boolean
}

function qualityDot(analyses: TurnAnalysis[], turn: number) {
  const analysis = analyses.find((a) => a.turn === turn)
  if (!analysis) return null
  const score = analysis.quality_score
  const color =
    score > 0.8
      ? "bg-emerald-400"
      : score >= 0.5
        ? "bg-amber-400"
        : "bg-red-400"
  return <span className={`inline-block w-2 h-2 rounded-full ${color} ml-2`} title={`Quality: ${score}`} />
}

function severityColor(severity: string) {
  switch (severity) {
    case "error": return "bg-red-100 text-red-700"
    case "warning": return "bg-amber-100 text-amber-700"
    default: return "bg-blue-100 text-blue-700"
  }
}

function categoryColor(category: string) {
  switch (category) {
    case "guardrails": return "bg-purple-100 text-purple-700"
    case "tool_call": return "bg-blue-100 text-blue-700"
    case "model_routing": return "bg-indigo-100 text-indigo-700"
    case "memory": return "bg-teal-100 text-teal-700"
    case "error": return "bg-red-100 text-red-700"
    case "agent": return "bg-slate-100 text-slate-700"
    default: return "bg-gray-100 text-gray-700"
  }
}

function ReasoningStepView({ step }: { step: AgentReasoningStep }) {
  const [expanded, setExpanded] = useState(false)

  if (step.is_final) {
    return (
      <div className="flex items-start gap-2 py-1">
        <Badge variant="secondary" className="bg-emerald-100 text-emerald-700 shrink-0">Final</Badge>
        <span className="text-sm text-muted-foreground truncate">{step.thought.slice(0, 120)}...</span>
      </div>
    )
  }

  if (!step.tool_call) return null

  return (
    <div className="border-l-2 border-muted pl-3 py-1.5">
      {step.thought && (
        <p className="text-xs text-muted-foreground italic mb-1">{step.thought.slice(0, 200)}</p>
      )}
      <div className="flex items-center gap-2">
        <Badge variant="outline" className="font-mono text-xs">{step.tool_call.name}</Badge>
        <span className="text-xs text-muted-foreground">
          ({Object.keys(step.tool_call.args).join(", ")})
        </span>
      </div>
      {step.tool_call.result && (
        <div className="mt-1">
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs text-blue-600 hover:underline"
          >
            {expanded ? "Hide result" : "Show result"}
          </button>
          {expanded && (
            <pre className="mt-1 text-xs bg-muted p-2 rounded-md overflow-x-auto max-h-48 overflow-y-auto whitespace-pre-wrap">
              {step.tool_call.result}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}

function ConversationTab({ data }: { data: SessionDetailResponse }) {
  return (
    <div className="space-y-3">
      {data.messages.length === 0 && (
        <p className="text-sm text-muted-foreground text-center py-8">No messages yet.</p>
      )}
      {data.messages.map((msg, i) => (
        <div
          key={i}
          className={`flex gap-3 ${msg.role === "user" ? "" : "flex-row-reverse"}`}
        >
          <div
            className={`max-w-[80%] rounded-lg p-3 text-sm ${
              msg.role === "user"
                ? "bg-muted"
                : "bg-primary/5"
            }`}
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-medium text-muted-foreground uppercase">
                {msg.role}
              </span>
              <span className="text-xs text-muted-foreground">Turn {msg.turn}</span>
              {msg.blocked && (
                <Badge variant="destructive" className="text-[10px] px-1.5 py-0">Blocked</Badge>
              )}
              {qualityDot(data.analyses, msg.turn)}
            </div>
            <p className="whitespace-pre-wrap">{msg.content}</p>
          </div>
        </div>
      ))}
    </div>
  )
}

function TracesTab({ traces }: { traces: EnrichedTrace[] }) {
  const [expandedTurn, setExpandedTurn] = useState<number | null>(null)

  if (traces.length === 0) {
    return <p className="text-sm text-muted-foreground text-center py-8">No traces recorded.</p>
  }

  return (
    <div className="space-y-2">
      {traces.map((trace) => (
        <Card key={trace.turn} className="overflow-hidden">
          <button
            onClick={() => setExpandedTurn(expandedTurn === trace.turn ? null : trace.turn)}
            className="w-full px-4 py-3 flex items-center justify-between text-sm hover:bg-muted/50 transition-colors"
          >
            <div className="flex items-center gap-3">
              <span className="font-medium">Turn {trace.turn}</span>
              <span className="text-muted-foreground">{trace.total_duration_ms.toFixed(0)}ms</span>
              <Badge variant="outline" className="text-xs">{trace.model_tier}</Badge>
              {trace.tool_calls.length > 0 && (
                <Badge variant="secondary" className="text-xs">
                  {trace.tool_calls.length} tool{trace.tool_calls.length !== 1 ? "s" : ""}
                </Badge>
              )}
              {trace.input_blocked && (
                <Badge variant="destructive" className="text-xs">Blocked</Badge>
              )}
            </div>
            <span className="text-muted-foreground">{expandedTurn === trace.turn ? "−" : "+"}</span>
          </button>
          {expandedTurn === trace.turn && (
            <div className="px-4 pb-4 border-t">
              <div className="mt-3 space-y-2">
                {trace.reasoning_steps.length === 0 && (
                  <p className="text-xs text-muted-foreground">No reasoning steps recorded.</p>
                )}
                {trace.reasoning_steps.map((step) => (
                  <ReasoningStepView key={step.step_index} step={step} />
                ))}
              </div>
              {trace.pipeline_steps.length > 0 && (
                <div className="mt-3 pt-3 border-t">
                  <p className="text-xs font-medium text-muted-foreground mb-2">Pipeline Steps</p>
                  <div className="flex flex-wrap gap-2">
                    {trace.pipeline_steps.map((ps, i) => (
                      <Badge key={i} variant="outline" className="text-xs">
                        {ps.name} ({ps.duration_ms.toFixed(0)}ms)
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </Card>
      ))}
    </div>
  )
}

function AnalysisTab({ analyses }: { analyses: TurnAnalysis[] }) {
  const flaggedAnalyses = analyses.filter((a) => a.flags.length > 0)

  if (analyses.length === 0) {
    return <p className="text-sm text-muted-foreground text-center py-8">No analysis data. Click "Analyze" to run.</p>
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4 text-sm">
        <span className="text-muted-foreground">
          {analyses.length} turns analyzed
        </span>
        <span className="text-muted-foreground">
          {flaggedAnalyses.length} with issues
        </span>
      </div>
      {flaggedAnalyses.length === 0 && (
        <p className="text-sm text-emerald-600 text-center py-4">All turns passed quality checks.</p>
      )}
      {flaggedAnalyses.map((a) => (
        <Card key={a.turn} className="p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="font-medium text-sm">Turn {a.turn}</span>
            <div className="flex items-center gap-2">
              <span className={`text-sm font-medium ${a.quality_score > 0.8 ? "text-emerald-600" : a.quality_score >= 0.5 ? "text-amber-600" : "text-red-600"}`}>
                {a.quality_score.toFixed(2)}
              </span>
              {a.tool_call_assessment && (
                <Badge variant="outline" className="text-xs">{a.tool_call_assessment}</Badge>
              )}
            </div>
          </div>
          {a.summary && <p className="text-sm text-muted-foreground mb-3">{a.summary}</p>}
          <div className="space-y-2">
            {a.flags.map((flag, i) => (
              <div key={i} className="flex items-start gap-2">
                <Badge className={`shrink-0 text-xs ${severityColor(flag.severity)}`}>
                  {flag.flag_type}
                </Badge>
                <div className="text-sm">
                  <p>{flag.description}</p>
                  {flag.evidence && (
                    <p className="text-xs text-muted-foreground mt-0.5 italic">"{flag.evidence}"</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  )
}

function EventsTab({ events }: { events: ObservabilityEvent[] }) {
  const [filter, setFilter] = useState<string | null>(null)
  const categories = [...new Set(events.map((e) => e.category))]
  const filtered = filter ? events.filter((e) => e.category === filter) : events

  if (events.length === 0) {
    return <p className="text-sm text-muted-foreground text-center py-8">No events recorded.</p>
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <Button
          variant={filter === null ? "default" : "outline"}
          size="sm"
          onClick={() => setFilter(null)}
        >
          All ({events.length})
        </Button>
        {categories.map((cat) => (
          <Button
            key={cat}
            variant={filter === cat ? "default" : "outline"}
            size="sm"
            onClick={() => setFilter(cat)}
          >
            {cat} ({events.filter((e) => e.category === cat).length})
          </Button>
        ))}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b text-left text-muted-foreground">
              <th className="pb-2 pr-3 font-medium">Time</th>
              <th className="pb-2 pr-3 font-medium">Category</th>
              <th className="pb-2 pr-3 font-medium">Event</th>
              <th className="pb-2 pr-3 font-medium">Turn</th>
              <th className="pb-2 pr-3 font-medium">Duration</th>
              <th className="pb-2 font-medium">Detail</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((event, i) => (
              <tr key={i} className="border-b last:border-0">
                <td className="py-2 pr-3 font-mono whitespace-nowrap">
                  {event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : "-"}
                </td>
                <td className="py-2 pr-3">
                  <Badge className={`text-[10px] ${categoryColor(event.category)}`}>
                    {event.category}
                  </Badge>
                </td>
                <td className="py-2 pr-3">{event.event_type}</td>
                <td className="py-2 pr-3">{event.turn || "-"}</td>
                <td className="py-2 pr-3">
                  {event.duration_ms ? `${event.duration_ms.toFixed(0)}ms` : "-"}
                </td>
                <td className="py-2 font-mono text-muted-foreground max-w-xs truncate">
                  {Object.keys(event.detail).length > 0 ? JSON.stringify(event.detail) : "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function SessionDetail({ data, onBack, onAnalyze, analyzing }: SessionDetailProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={onBack}>
            &larr; Back
          </Button>
          <h2 className="font-semibold text-lg font-mono">{data.session_id}</h2>
        </div>
        <Button variant="outline" size="sm" onClick={onAnalyze} disabled={analyzing}>
          {analyzing ? "Analyzing..." : "Analyze"}
        </Button>
      </div>

      <Tabs defaultValue="conversation">
        <TabsList>
          <TabsTrigger value="conversation">Conversation</TabsTrigger>
          <TabsTrigger value="traces">
            Traces ({data.traces.length})
          </TabsTrigger>
          <TabsTrigger value="analysis">
            Analysis ({data.analyses.filter((a) => a.flags.length > 0).length} flagged)
          </TabsTrigger>
          <TabsTrigger value="events">
            Events ({data.events.length})
          </TabsTrigger>
        </TabsList>
        <TabsContent value="conversation" className="mt-4">
          <ConversationTab data={data} />
        </TabsContent>
        <TabsContent value="traces" className="mt-4">
          <TracesTab traces={data.traces} />
        </TabsContent>
        <TabsContent value="analysis" className="mt-4">
          <AnalysisTab analyses={data.analyses} />
        </TabsContent>
        <TabsContent value="events" className="mt-4">
          <EventsTab events={data.events} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
