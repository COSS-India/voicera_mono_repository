"use client"

import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import {
  isAllowedSpokenMessageChar,
  sanitizeSpokenMessageInput,
} from "@/lib/spoken-message"
import { cn } from "@/lib/utils"
import type { ComponentProps } from "react"

function useSpokenMessageHandlers(
  value: string,
  onChange: (value: string) => void
) {
  const commitValue = (next: string) => {
    onChange(sanitizeSpokenMessageInput(next))
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    if (
      e.key.length === 1 &&
      !e.ctrlKey &&
      !e.metaKey &&
      !e.altKey &&
      !e.nativeEvent.isComposing &&
      !isAllowedSpokenMessageChar(e.key)
    ) {
      e.preventDefault()
    }
  }

  const handlePaste = (
    e: React.ClipboardEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    e.preventDefault()
    const pasted = e.clipboardData.getData("text")
    const input = e.currentTarget
    const start = input.selectionStart ?? value.length
    const end = input.selectionEnd ?? value.length
    commitValue(value.slice(0, start) + pasted + value.slice(end))
  }

  return { commitValue, handleKeyDown, handlePaste }
}

type SpokenMessageInputProps = Omit<
  ComponentProps<typeof Input>,
  "value" | "onChange"
> & {
  value: string
  onChange: (value: string) => void
}

export function SpokenMessageInput({
  value,
  onChange,
  className,
  onKeyDown,
  onPaste,
  ...props
}: SpokenMessageInputProps) {
  const { commitValue, handleKeyDown, handlePaste } = useSpokenMessageHandlers(
    value,
    onChange
  )

  return (
    <Input
      {...props}
      value={value}
      className={cn(className)}
      onChange={(e) => commitValue(e.target.value)}
      onKeyDown={(e) => {
        handleKeyDown(e)
        onKeyDown?.(e)
      }}
      onPaste={(e) => {
        handlePaste(e)
        onPaste?.(e)
      }}
    />
  )
}

type SpokenMessageTextareaProps = Omit<
  ComponentProps<typeof Textarea>,
  "value" | "onChange"
> & {
  value: string
  onChange: (value: string) => void
}

export function SpokenMessageTextarea({
  value,
  onChange,
  className,
  onKeyDown,
  onPaste,
  ...props
}: SpokenMessageTextareaProps) {
  const { commitValue, handleKeyDown, handlePaste } = useSpokenMessageHandlers(
    value,
    onChange
  )

  return (
    <Textarea
      {...props}
      value={value}
      className={cn(className)}
      onChange={(e) => commitValue(e.target.value)}
      onKeyDown={(e) => {
        handleKeyDown(e)
        onKeyDown?.(e)
      }}
      onPaste={(e) => {
        handlePaste(e)
        onPaste?.(e)
      }}
    />
  )
}
