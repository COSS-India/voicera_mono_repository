"use client"

import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import type { ComponentProps } from "react"

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
  ...props
}: SpokenMessageInputProps) {
  return (
    <Input
      {...props}
      value={value}
      className={cn(className)}
      onChange={(e) => onChange(e.target.value)}
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
  ...props
}: SpokenMessageTextareaProps) {
  return (
    <Textarea
      {...props}
      value={value}
      className={cn(className)}
      onChange={(e) => onChange(e.target.value)}
    />
  )
}
