import { motion } from "framer-motion"
import type { ConversationPhase } from "@/types"

const chipsByPhase: Record<ConversationPhase, string[]> = {
  intake: [
    "Homework struggles",
    "Morning routines are tough",
    "Emotional meltdowns",
    "Focus and attention",
  ],
  strategy: [
    "Tell me more about that",
    "What else can I try?",
    "That sounds helpful",
    "Can we try something different?",
  ],
  progress: [
    "It's going well!",
    "We're still struggling",
    "I have a question",
    "Let's set a new goal",
  ],
  followup: [
    "Check in on our goals",
    "Try a new strategy",
    "Things have changed",
    "I need more support",
  ],
}

interface Props {
  phase: ConversationPhase
  onSelect: (text: string) => void
  visible: boolean
}

export function QuickReplyChips({ phase, onSelect, visible }: Props) {
  if (!visible) return null

  const chips = chipsByPhase[phase] || chipsByPhase.intake

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -4 }}
      transition={{ duration: 0.2 }}
      className="flex flex-wrap gap-2 px-4 pb-2"
    >
      {chips.map((text, i) => (
        <motion.button
          key={text}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2, delay: i * 0.05 }}
          onClick={() => onSelect(text)}
          className="rounded-full border border-primary/30 bg-transparent px-4 py-2 text-sm text-primary transition-colors duration-150 hover:bg-primary/10 active:scale-[0.98]"
        >
          {text}
        </motion.button>
      ))}
    </motion.div>
  )
}
