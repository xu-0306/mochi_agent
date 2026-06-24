'use client'

import { create } from 'zustand'
import {
  createSession as createSessionApi,
  deleteSession as deleteSessionApi,
  fetchSession,
  fetchSessions,
  forkSession as forkSessionApi,
  renameSession as renameSessionApi,
  updateSessionProject as updateSessionProjectApi,
  type SessionSecurityOverride,
  type SessionWorkflowState,
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
  projectId: string | null
  workflow: SessionWorkflowState | null
  securityOverride: SessionSecurityOverride | null
  isDraft: boolean
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
  createDraftSession: (projectId: string | null) => string
  createSession: (projectId?: string | null) => string
  materializeDraftSession: (draftId: string) => Promise<string>
  forkSessionFromTurn: (
    sessionId: string,
    turnId: string,
    projectId?: string | null
  ) => Promise<string>
  pinSession: (id: string) => void
  deleteSession: (id: string) => Promise<void>
  renameSession: (id: string, title: string) => Promise<void>
  updateLastMessage: (id: string, message: string) => void
  moveSessionToProject: (id: string, projectId: string | null) => Promise<void>
}

function extractLastMessage(detail: SessionDetail): string {
  const latestMessageEvent = [...detail.events]
    .reverse()
    .find((event) => event.type === 'message')

  if (!latestMessageEvent) {
    return ''
  }

  if (typeof latestMessageEvent.content === 'string' && latestMessageEvent.content.trim().length > 0) {
    return latestMessageEvent.content
  }

  const attachments = Array.isArray((latestMessageEvent as { attachments?: unknown }).attachments)
    ? (latestMessageEvent as { attachments?: Array<{ name?: unknown }> }).attachments ?? []
    : []
  const names = attachments
    .map((attachment) => (typeof attachment?.name === 'string' ? attachment.name.trim() : ''))
    .filter((name) => name.length > 0)

  return names.join(', ')
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
    projectId: summary.projectId,
    workflow: summary.workflow,
    securityOverride: summary.security_override,
    isDraft: false,
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
    projectId: detail.projectId,
    workflow: detail.workflow,
    securityOverride: detail.security_override,
    isDraft: false,
  }
}

function buildDraftSession(projectId: string | null): Session {
  const now = new Date()
  return {
    id: `draft-${Date.now()}`,
    title: DEFAULT_SESSION_TITLE,
    lastMessage: '',
    lastMessageAt: now,
    source: 'web',
    isPinned: false,
    messageCount: 0,
    projectId,
    workflow: null,
    securityOverride: null,
    isDraft: true,
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
      const persistedSessions = summaries.map(normalizeSummary)

      set((state) => {
        const draftSessions = state.sessions.filter((session) => session.isDraft)
        const sessions = [...draftSessions, ...persistedSessions]
        const currentSessionId =
          state.currentSessionId && sessions.some((session) => session.id === state.currentSessionId)
            ? state.currentSessionId
            : (sessions[0]?.id ?? null)

        return {
          sessions,
          currentSessionId,
          currentSessionDetail:
            state.currentSessionDetail &&
            sessions.some((session) => session.id === state.currentSessionDetail?.id)
              ? state.currentSessionDetail
              : null,
          isLoading: false,
          hasLoaded: true,
          error: null,
        }
      })

      const currentSessionId = get().currentSessionId
      const currentSession = get().sessions.find((session) => session.id === currentSessionId)
      if (currentSessionId && currentSession && !currentSession.isDraft) {
        void get().selectSession(currentSessionId)
      }
    } catch (error: unknown) {
      set((state) => ({
        sessions: state.sessions.filter((session) => session.isDraft),
        currentSessionId: state.currentSessionId,
        currentSessionDetail: state.currentSessionDetail,
        isLoading: false,
        isLoadingDetail: false,
        hasLoaded: true,
        error: error instanceof Error ? error.message : 'Failed to load sessions',
      }))
    }
  },

  selectSession: async (id) => {
    const target = get().sessions.find((session) => session.id === id)
    set({
      currentSessionId: id,
      isLoadingDetail: target?.isDraft ? false : true,
      error: null,
      currentSessionDetail:
        target?.isDraft
          ? {
              id: target.id,
              title: target.title,
              createdAt: target.lastMessageAt.toISOString(),
              updatedAt: target.lastMessageAt.toISOString(),
              eventCount: 0,
              projectId: target.projectId,
              workflow: target.workflow,
              security_override: target.securityOverride,
              events: [],
            }
          : get().currentSessionDetail,
    })

    if (target?.isDraft) {
      return
    }

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

  createDraftSession: (projectId) => {
    const draft = buildDraftSession(projectId)
    set((state) => ({
      sessions: [draft, ...state.sessions.filter((session) => !session.isDraft)],
      currentSessionId: draft.id,
      currentSessionDetail: {
        id: draft.id,
        title: draft.title,
        createdAt: draft.lastMessageAt.toISOString(),
        updatedAt: draft.lastMessageAt.toISOString(),
        eventCount: 0,
        projectId: draft.projectId,
        workflow: draft.workflow,
        security_override: draft.securityOverride,
        events: [],
      },
      error: null,
    }))
    return draft.id
  },

  createSession: (projectId = null) => get().createDraftSession(projectId),

  materializeDraftSession: async (draftId) => {
    const draft = get().sessions.find((session) => session.id === draftId)
    if (!draft) {
      return draftId
    }
    if (!draft.isDraft) {
      return draft.id
    }

    const created = await createSessionApi(undefined, draft.projectId)
    const detail = await fetchSession(created.id)

    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.id === draftId
          ? {
              ...mergeSession(
                {
                  ...normalizeSummary(created),
                  id: created.id,
                  lastMessage: session.lastMessage,
                  lastMessageAt: session.lastMessageAt,
                  isPinned: session.isPinned,
                  projectId: draft.projectId,
                  workflow: session.workflow,
                  isDraft: false,
                },
                detail
              ),
              id: created.id,
              lastMessage: session.lastMessage,
              lastMessageAt: session.lastMessageAt,
            }
          : session
      ),
      currentSessionId: state.currentSessionId === draftId ? created.id : state.currentSessionId,
      currentSessionDetail:
        state.currentSessionId === draftId
          ? {
              ...detail,
              title: sessionTitleForMaterialized(detail.title, draft.title),
            }
          : state.currentSessionDetail,
      error: null,
    }))

    return created.id
  },

  forkSessionFromTurn: async (sessionId, turnId, projectId) => {
    const created = await forkSessionApi({ sessionId, turnId, projectId })
    const detail = await fetchSession(created.id)
    const normalized = normalizeSummary(created)
    const merged = mergeSession(normalized, detail)

    set((state) => ({
      sessions: [
        merged,
        ...state.sessions.filter((session) => session.id !== created.id && !session.isDraft),
        ...state.sessions.filter((session) => session.isDraft),
      ],
      currentSessionId: created.id,
      currentSessionDetail: detail,
      error: null,
    }))

    return created.id
  },

  pinSession: (id) =>
    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.id === id ? { ...session, isPinned: !session.isPinned } : session
      ),
    })),

  deleteSession: async (id) => {
    const target = get().sessions.find((session) => session.id === id)
    if (!target) {
      return
    }

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
      if (!target.isDraft) {
        await deleteSessionApi(id)
      }
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
    const target = previous.sessions.find((session) => session.id === id)
    if (!target) {
      return
    }

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

    if (target.isDraft) {
      return
    }

    try {
      await renameSessionApi(id, trimmedTitle)
      const detail = await fetchSession(id)
      set((state) => ({
        sessions: state.sessions.map((session) =>
          session.id === id ? mergeSession(session, detail) : session
        ),
        currentSessionDetail:
          state.currentSessionDetail?.id === id ? detail : state.currentSessionDetail,
        error: null,
      }))
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
              title:
                session.messageCount === 0 && !session.lastMessage
                  ? message.slice(0, 80)
                  : session.title,
            }
          : session
      ),
    })),

  moveSessionToProject: async (id, projectId) => {
    const previous = get()
    const target = previous.sessions.find((session) => session.id === id)
    if (!target) {
      return
    }

    set((state) => ({
      sessions: state.sessions.map((session) =>
        session.id === id ? { ...session, projectId } : session
      ),
      currentSessionDetail:
        state.currentSessionDetail?.id === id
          ? { ...state.currentSessionDetail, projectId }
          : state.currentSessionDetail,
      error: null,
    }))

    if (target.isDraft) {
      return
    }

    try {
      await updateSessionProjectApi(id, projectId)
    } catch (error: unknown) {
      set({
        sessions: previous.sessions,
        currentSessionId: previous.currentSessionId,
        currentSessionDetail: previous.currentSessionDetail,
        error: error instanceof Error ? error.message : 'Failed to move session',
      })
    }
  },
}))

function sessionTitleForMaterialized(nextTitle: string, draftTitle: string): string {
  return nextTitle && nextTitle !== DEFAULT_SESSION_TITLE ? nextTitle : draftTitle
}

export const useSessionStore = sessionStore
