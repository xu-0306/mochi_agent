'use client'

import { create } from 'zustand'
import type { Message } from '@/lib/chat'

function toTimestampValue(value: Date | undefined): number | null {
  return value instanceof Date ? value.getTime() : null
}

function buildMessageSignature(message: Message): string {
  return JSON.stringify({
    id: message.id,
    type: message.type,
    content: message.content,
    timestamp: toTimestampValue(message.timestamp),
    eventType: message.eventType ?? null,
    turnKey: message.turnKey ?? null,
    turnId: message.turnId ?? null,
    errorCode: message.errorCode ?? null,
    isStreaming: message.isStreaming ?? false,
    inlineReasoningStepId: message.inlineReasoningStepId ?? null,
    tokenStats: message.tokenStats ?? null,
    attachments: (message.attachments ?? []).map((attachment) => ({
      id: attachment.id ?? null,
      name: attachment.name,
      path: attachment.path,
      size: attachment.size ?? null,
      contentType: attachment.contentType ?? null,
      source: attachment.source ?? null,
      lineStart: attachment.lineStart ?? null,
      lineEnd: attachment.lineEnd ?? null,
      quote: attachment.quote ?? null,
      note: attachment.note ?? null,
    })),
    goalCard: message.goalCard
      ? {
          kind: message.goalCard.kind,
          label: message.goalCard.label,
          objective: message.goalCard.objective,
          executionMode: message.goalCard.executionMode,
          protocolId: message.goalCard.protocolId ?? null,
          models: message.goalCard.models,
          roleSummary: message.goalCard.roleSummary ?? null,
          runtimeMode: message.goalCard.runtimeMode ?? null,
          riskNote: message.goalCard.riskNote ?? null,
          goalId: message.goalCard.goalId ?? null,
          status: message.goalCard.status ?? null,
          superseded: message.goalCard.superseded ?? null,
        }
      : null,
    reasoningSteps: (message.reasoningSteps ?? []).map((step) => ({
      id: step.id,
      type: step.type,
      content: step.content,
      timestamp: toTimestampValue(step.timestamp),
      source: step.source ?? null,
      toolName: step.toolName ?? null,
      toolArgs: step.toolArgs ?? null,
      toolResult: step.toolResult ?? null,
      toolMeta: step.toolMeta ?? null,
      toolCallId: step.toolCallId ?? null,
      toolError: step.toolError ?? null,
      errorCode: step.errorCode ?? null,
      status: step.status ?? null,
    })),
    reasoningBuffer: message.reasoningBuffer ?? null,
  })
}

function areMessagesEquivalent(current: Message[] | undefined, next: Message[]): boolean {
  if (current === next) {
    return true
  }
  if (!current || current.length !== next.length) {
    return false
  }
  return current.every(
    (message, index) => buildMessageSignature(message) === buildMessageSignature(next[index])
  )
}

interface ChatRuntimeStore {
  messagesBySessionId: Record<string, Message[]>
  streamingSessionId: string | null
  activeAbortController: AbortController | null

  setSessionMessages: (sessionId: string, messages: Message[]) => void
  updateSessionMessages: (sessionId: string, updater: (messages: Message[]) => Message[]) => void
  hydrateSessionMessages: (sessionId: string, messages: Message[]) => void
  clearSessionMessages: (sessionId: string) => void
  startStreaming: (sessionId: string, controller: AbortController) => void
  finishStreaming: (sessionId: string) => void
  abortStreaming: () => void
}

export const useChatRuntimeStore = create<ChatRuntimeStore>((set, get) => ({
  messagesBySessionId: {},
  streamingSessionId: null,
  activeAbortController: null,

  setSessionMessages: (sessionId, messages) =>
    set((state) => {
      if (areMessagesEquivalent(state.messagesBySessionId[sessionId], messages)) {
        return state
      }
      return {
        messagesBySessionId: {
          ...state.messagesBySessionId,
          [sessionId]: messages,
        },
      }
    }),

  updateSessionMessages: (sessionId, updater) =>
    set((state) => {
      const current = state.messagesBySessionId[sessionId] ?? []
      const next = updater(current)
      if (areMessagesEquivalent(current, next)) {
        return state
      }
      return {
        messagesBySessionId: {
          ...state.messagesBySessionId,
          [sessionId]: next,
        },
      }
    }),

  hydrateSessionMessages: (sessionId, messages) =>
    set((state) => {
      const existing = state.messagesBySessionId[sessionId]
      if (existing && existing.length > 0) {
        return state
      }
      if (areMessagesEquivalent(existing, messages)) {
        return state
      }
      return {
        messagesBySessionId: {
          ...state.messagesBySessionId,
          [sessionId]: messages,
        },
      }
    }),

  clearSessionMessages: (sessionId) =>
    set((state) => {
      const next = { ...state.messagesBySessionId }
      delete next[sessionId]
      return { messagesBySessionId: next }
    }),

  startStreaming: (sessionId, controller) =>
    set({
      streamingSessionId: sessionId,
      activeAbortController: controller,
    }),

  finishStreaming: (sessionId) =>
    set((state) => (
      state.streamingSessionId === sessionId
        ? { streamingSessionId: null, activeAbortController: null }
        : state
    )),

  abortStreaming: () => {
    get().activeAbortController?.abort()
  },
}))
