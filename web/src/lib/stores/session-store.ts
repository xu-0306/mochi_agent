'use client'

import { create } from 'zustand'
import {
  createSession as createSessionApi,
  deleteSession as deleteSessionApi,
  fetchSession,
  fetchSessions,
  renameSession as renameSessionApi,
  type SessionDetail,
  type SessionSummary,
} from '@/lib/api'

const DEFAULT_SESSION_TITLE = 'New chat'

export type ChannelSource = 'web' | 'cli' | 'discord' | 'telegram'

export interface Session {
  id: string
  title: string
  lastMessage: string
  lastMessageAt: Date
  source: ChannelSource
  isPinned: boolean
  messageCount: number
}

interface SessionStore {
  sessions: Session[]
  currentSessionId: string | null
  currentSessionDetail: SessionDetail | null
  isLoading: boolean
  isLoadingDetail: boolean
  hasLoaded: boolean
  error: string | null

  loadSessions: () => Promise<void>
  selectSession: (id: string) => Promise<void>
  setCurrentSession: (id: string) => void
  createSession: () => string
  pinSession: (id: string) => void
  deleteSession: (id: string) => Promise<void>
  renameSession: (id: string, title: string) => Promise<void>
  updateLastMessage: (id: string, message: string) => void
}

function extractLastMessage(detail: SessionDetail): string {
  const latestMessageEvent = [...detail.events]
    .reverse()
    .find(
      (event) =>
        event.type === 'message' &&
        typeof event.content === 'string' &&
        event.content.trim().length > 0
    )

  return latestMessageEvent?.content ?? ''
}

function extractTitle(detail: SessionDetail): string {
  return detail.title || detail.id
}

function extractLastMessageAt(detail: SessionDetail, fallback: string): Date {
  const latestTimestamp = [...detail.events]
    .reverse()
    .map((event) => event.timestamp)
    .find((timestamp): timestamp is string => typeof timestamp === 'string')

  return new Date(latestTimestamp ?? fallback)
}

function inferSource(sessionId: string): ChannelSource {
  if (sessionId.startsWith('discord-')) {
    return 'discord'
  }
  if (sessionId.startsWith('telegram-')) {
    return 'telegram'
  }
  if (sessionId.startsWith('cli-') || sessionId.startsWith('voice-cli')) {
    return 'cli'
  }
  return 'web'
}

function normalizeSummary(summary: SessionSummary): Session {
  return {
    id: summary.id,
    title: summary.title,
    lastMessage: '',
    lastMessageAt: new Date(summary.updatedAt),
    source: inferSource(summary.id),
    isPinned: false,
    messageCount: summary.eventCount,
  }
}

function mergeSession(target: Session, detail: SessionDetail): Session {
  return {
    ...target,
    title: extractTitle(detail),
    lastMessage: extractLastMessage(detail),
    lastMessageAt: extractLastMessageAt(detail, detail.updatedAt),
    source: inferSource(detail.id),
    messageCount: detail.eventCount,
  }
}

const sessionStore = create<SessionStore>((set, get) => ({
  sessions: [],
  currentSessionId: null,
  currentSessionDetail: null,
  isLoading: false,
  isLoadingDetail: false,
  hasLoaded: false,
  error: null,

  loadSessions: async () => {
    if (get().isLoading) {
      return
    }

    set({ isLoading: true, error: null })

    try {
      const summaries = await fetchSessions()
      const sessions = summaries.map(normalizeSummary)

      set((state) => ({
        sessions,
        currentSessionId:
          state.currentSessionId && sessions.some((session) => session.id === state.currentSessionId)
            ? state.currentSessionId
            : (sessions[0]?.id ?? null),
        currentSessionDetail:
          state.currentSessionDetail &&
          sessions.some((session) => session.id === state.currentSessionDetail?.id)
            ? state.currentSessionDetail
            : null,
        isLoading: false,
        hasLoaded: true,
        error: null,
      }))

      const currentSessionId = get().currentSessionId
      if (currentSessionId) {
        void get().selectSession(currentSessionId)
      }
    } catch (error: unknown) {
      set({
        sessions: [],
        currentSessionId: null,
        currentSessionDetail: null,
        isLoading: false,
        isLoadingDetail: false,
        hasLoaded: true,
        error: error instanceof Error ? error.message : 'Failed to load sessions',
      })
    }
  },

  selectSession: async (id) => {
    set({ currentSessionId: id, isLoadingDetail: true, error: null })

    try {
      const detail = await fetchSession(id)
      if (get().currentSessionId !== id) {
        return
      }
      set((state) => ({
        sessions: state.sessions.map((session) =>
          session.id === id ? mergeSession(session, detail) : session
        ),
        currentSessionDetail: detail,
        isLoadingDetail: false,
        error: null,
      }))
    } catch (error: unknown) {
      if (get().currentSessionId !== id) {
        return
      }
      set({
        currentSessionDetail: null,
        isLoadingDetail: false,
        error: error instanceof Error ? error.message : 'Failed to load session detail',
      })
    }
  },

  setCurrentSession: (id) => {
    set({ currentSessionId: id })
    void get().selectSession(id)
  },

  createSession: () => {
    const optimisticId = `pending-${Date.now()}`
    const optimisticSession: Session = {
      id: optimisticId,
      title: DEFAULT_SESSION_TITLE,
      lastMessage: '',
      lastMessageAt: new Date(),
      source: 'web',
      isPinned: false,
      messageCount: 0,
    }

    set((state) => ({
      sessions: [optimisticSession, ...state.sessions],
      currentSessionId: optimisticId,
      currentSessionDetail: {
        id: optimisticId,
        title: DEFAULT_SESSION_TITLE,
        createdAt: optimisticSession.lastMessageAt.toISOString(),
        updatedAt: optimisticSession.lastMessageAt.toISOString(),
        eventCount: 0,
        events: [],
      },
      error: null,
    }))

    void (async () => {
      try {
        const created = await createSessionApi(optimisticId)
        const createdSession = normalizeSummary(created)

        set((state) => ({
          sessions: state.sessions.map((session) =>
            session.id === optimisticId
              ? {
                  ...createdSession,
                  title: DEFAULT_SESSION_TITLE,
                  isPinned: session.isPinned,
                  lastMessage: session.lastMessage,
                  lastMessageAt: session.lastMessageAt,
                }
              : session
          ),
          currentSessionId:
            state.currentSessionId === optimisticId ? createdSession.id : state.currentSessionId,
          currentSessionDetail:
            state.currentSessionId === optimisticId
              ? {
                  id: created.id,
                  title: DEFAULT_SESSION_TITLE,
                  createdAt: created.createdAt,
                  updatedAt: created.updatedAt,
                  eventCount: 0,
                  events: [],
                }
              : state.currentSessionDetail,
          error: null,
        }))
      } catch (error: unknown) {
        set((state) => ({
          sessions: state.sessions.filter((session) => session.id !== optimisticId),
          currentSessionId:
            state.currentSessionId === optimisticId ? (state.sessions[1]?.id ?? null) : state.currentSessionId,
          currentSessionDetail:
            state.currentSessionId === optimisticId ? null : state.currentSessionDetail,
          error: error instanceof Error ? error.message : 'Failed to create session',
        }))
      }
    })()

    return optimisticId
  },

  pinSession: (id) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.id === id ? { ...session, isPinned: !session.isPinned } : session
      ),
    })),

  deleteSession: async (id) => {
    const previous = get()
    let nextCurrentSessionId: string | null = null
    set((state) => {
      const nextSessions = state.sessions.filter((session) => session.id !== id)
      nextCurrentSessionId =
        state.currentSessionId === id ? (nextSessions[0]?.id ?? null) : state.currentSessionId
      return {
        sessions: nextSessions,
        currentSessionId: nextCurrentSessionId,
        currentSessionDetail: state.currentSessionId === id ? null : state.currentSessionDetail,
        error: null,
      }
    })

    try {
      await deleteSessionApi(id)
    } catch (error: unknown) {
      set({
        sessions: previous.sessions,
        currentSessionId: previous.currentSessionId,
        currentSessionDetail: previous.currentSessionDetail,
        error: error instanceof Error ? error.message : 'Failed to delete session',
      })
      return
    }

    if (nextCurrentSessionId && previous.currentSessionId === id) {
      void get().selectSession(nextCurrentSessionId)
    }
  },

  renameSession: async (id, title) => {
    const trimmedTitle = title.trim()
    if (!trimmedTitle) {
      return
    }

    const previous = get()
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.id === id ? { ...session, title: trimmedTitle } : session
      ),
      currentSessionDetail:
        state.currentSessionDetail?.id === id
          ? { ...state.currentSessionDetail, title: trimmedTitle }
          : state.currentSessionDetail,
      error: null,
    }))

    try {
      await renameSessionApi(id, trimmedTitle)
      const detail = await fetchSession(id)
      if (get().currentSessionId === id) {
        set((state) => ({
          sessions: state.sessions.map((session) =>
            session.id === id ? mergeSession(session, detail) : session
          ),
          currentSessionDetail: detail,
          error: null,
        }))
      }
    } catch (error: unknown) {
      set({
        sessions: previous.sessions,
        currentSessionId: previous.currentSessionId,
        currentSessionDetail: previous.currentSessionDetail,
        error: error instanceof Error ? error.message : 'Failed to rename session',
      })
    }
  },

  updateLastMessage: (id, message) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.id === id
          ? {
              ...session,
              lastMessage: message,
              lastMessageAt: new Date(),
              title: session.messageCount === 0 && !session.lastMessage
                ? message.slice(0, 80)
                : session.title,
            }
          : session
      ),
    })),
}))

export const useSessionStore = sessionStore

export function getPinnedSessions(sessions: Session[]): Session[] {
  return sessions.filter((session) => session.isPinned)
}

export function getTodaySessions(sessions: Session[]): Session[] {
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  return sessions.filter((session) => !session.isPinned && session.lastMessageAt >= today)
}

export function getThisWeekSessions(sessions: Session[]): Session[] {
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000)
  return sessions.filter(
    (session) =>
      !session.isPinned &&
      session.lastMessageAt < today &&
      session.lastMessageAt >= weekAgo
  )
}

export function getOlderSessions(sessions: Session[]): Session[] {
  const weekAgo = new Date()
  weekAgo.setHours(0, 0, 0, 0)
  weekAgo.setDate(weekAgo.getDate() - 7)
  return sessions.filter((session) => !session.isPinned && session.lastMessageAt < weekAgo)
}
