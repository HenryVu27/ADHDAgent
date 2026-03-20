import { motion } from "framer-motion"

interface Props {
  suggestions?: string[]
  onSelect: (text: string) => void
  visible: boolean
}

export function QuickReplyChips({ suggestions, onSelect, visible }: Props) {
  if (!visible || !suggestions || suggestions.length === 0) return null

  const chips = suggestions

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
