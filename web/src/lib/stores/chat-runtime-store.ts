'use client'

import { create } from 'zustand'
import type { Message } from '@/lib/chat'

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
    set((state) => ({
      messagesBySessionId: {
        ...state.messagesBySessionId,
        [sessionId]: messages,
      },
    })),

  updateSessionMessages: (sessionId, updater) =>
    set((state) => ({
      messagesBySessionId: {
        ...state.messagesBySessionId,
        [sessionId]: updater(state.messagesBySessionId[sessionId] ?? []),
      },
    })),

  hydrateSessionMessages: (sessionId, messages) =>
    set((state) => {
      const existing = state.messagesBySessionId[sessionId]
      if (existing && existing.length > 0) {
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
