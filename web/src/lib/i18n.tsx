'use client'

import * as React from 'react'
import { DEFAULT_CODE_THEME, normalizeCodeTheme, type UICodeTheme } from '@/lib/code-theme'

export type UILanguage = 'zh-TW' | 'en'
export type UILanguageMode = 'auto' | UILanguage
export type UIFontSize = 'compact' | 'default' | 'large'
export type UITimezone = 'auto' | string
export type UIAppearanceMode = 'system' | 'dark' | 'light'
type UIAppearance = 'dark' | 'light'

export const AUTO_LANGUAGE = 'auto' as const
export const AUTO_TIMEZONE = 'auto' as const
export const SYSTEM_APPEARANCE = 'system' as const
const DEFAULT_LANGUAGE: UILanguage = 'en'
const DEFAULT_LANGUAGE_MODE: UILanguageMode = AUTO_LANGUAGE
const DEFAULT_FONT_SIZE: UIFontSize = 'default'
const DEFAULT_APPEARANCE: UIAppearance = 'dark'
const DEFAULT_APPEARANCE_MODE: UIAppearanceMode = SYSTEM_APPEARANCE
const THEME_COLOR_BY_APPEARANCE: Record<UIAppearance, string> = {
  dark: '#0B0B0F',
  light: '#FBFBFC',
}

const LANGUAGE_LOCALE: Record<UILanguage, string> = {
  'zh-TW': 'zh-TW',
  en: 'en-US',
}

type TranslationKey = string
type TranslationValues = Record<string, string | number | boolean | null | undefined>

type I18nContextValue = {
  languageMode: UILanguageMode
  setLanguageMode: (mode: UILanguageMode) => void
  language: UILanguage
  setLanguage: (language: UILanguage) => void
  locale: string
  fontSize: UIFontSize
  setFontSize: (fontSize: UIFontSize) => void
  appearanceMode: UIAppearanceMode
  setAppearanceMode: (mode: UIAppearanceMode) => void
  appearance: UIAppearance
  codeTheme: UICodeTheme
  setCodeTheme: (theme: UICodeTheme) => void
  timezone: UITimezone
  setTimezone: (timezone: UITimezone) => void
  resolvedTimeZone: string | undefined
  t: (key: TranslationKey, values?: TranslationValues) => string
}

const STORAGE_KEY = 'mochi.ui.preferences.v1'

function interpolate(message: string, values?: TranslationValues): string {
  if (!values) {
    return message
  }

  return message.replace(/\{(\w+)\}/g, (match, key: string) => {
    const value = values[key]
    return value === undefined || value === null ? match : String(value)
  })
}

const messages: Record<UILanguage, Record<TranslationKey, string>> = {
  'zh-TW': {
    'common.available': '可用',
    'common.cancel': '取消',
    'common.close': '關閉',
    'common.configured': '已設定',
    'common.disabled': '已停用',
    'common.enabled': '已啟用',
    'common.fields': '{count} 個欄位',
    'common.items': '{count} 個項目',
    'common.listSummary': '{items} 等 {count} 筆',
    'common.loading': '載入中…',
    'common.none': '無',
    'common.no': '否',
    'common.notConfigured': '未設定',
    'common.notReported': '未回報',
    'common.notSet': '未設定',
    'common.refresh': '重新整理',
    'common.reported': '已回報',
    'common.unavailable': '不可用',
    'common.unknown': '未知',
    'common.yes': '是',
    'chat.disclaimer': 'Mochi 可能犯錯，重要資訊請自行核實',
    'chat.emptyAssistantResponse': '代理沒有回傳任何內容。',
    'chat.input.attachFile': '上傳檔案',
    'chat.input.currentModel': '目前模型',
    'chat.input.placeholder': '輸入訊息… (Enter 送出，Shift+Enter 換行)',
    'chat.input.send': '送出 (Enter)',
    'chat.input.voice': '語音輸入 (⌘⇧V)',
    'chat.loadingLocalModel': '正在載入本地模型，可能需要一段時間…',
    'chat.loadingSession': '正在載入對話紀錄…',
    'chat.moreOptions': '更多選項',
    'chat.newChat': '新對話',
    'chat.modelSwitchFailed': '模型切換失敗',
    'chat.requestFailed': '無法完成本次對話請求',
    'chat.settingsShortcut': '設定 (⌘,)',
    'chat.system.ready': 'Mochi WebGUI 已就緒，送出訊息後會顯示推理、工具呼叫與最終回答事件。',
    'chat.thinking': '思考中',
    'chat.voice.assistant': '助理回覆',
    'chat.voice.close': '關閉',
    'chat.voice.deviceUnknown': '未識別麥克風',
    'chat.voice.interrupt': '中斷',
    'chat.voice.microphone': '麥克風',
    'chat.voice.noInputDetected': '尚未偵測到麥克風輸入訊號，請檢查瀏覽器權限與目前選用的錄音裝置。',
    'chat.voice.phase.connecting': '連線中',
    'chat.voice.phase.error': '錯誤',
    'chat.voice.phase.idle': '待命',
    'chat.voice.phase.listening': '聆聽中',
    'chat.voice.phase.ready': '就緒',
    'chat.voice.phase.synthesizing': '語音合成中',
    'chat.voice.phase.thinking': '思考中',
    'chat.voice.phase.transcribing': '語音轉寫中',
    'chat.voice.record': '開始錄音',
    'chat.voice.signalDetected': '已收到麥克風訊號',
    'chat.voice.signalWaiting': '等待麥克風訊號',
    'chat.voice.startPrompt': '開始錄音後即可說話。',
    'chat.voice.stop': '停止',
    'chat.voice.title': '語音對話',
    'chat.voice.transcription': '轉寫結果',
    'chat.voice.vadDetected': '已偵測到語音',
    'chat.voice.waiting': '等待回覆中。',
    'chat.tool.args': '參數',
    'chat.tool.error': '錯誤',
    'chat.tool.failed': '失敗',
    'chat.tool.result': '結果',
    'chat.tool.running': '執行中…',
    'errors.channelsStatusUnavailable': '後端未提供 /v1/channels 狀態端點',
    'errors.chatApiUnavailable': 'Chat API client 不可用',
    'errors.filesystemImportUnavailable': 'Filesystem import API client 不可用',
    'errors.modelConfigureApiUnavailable': 'Model configure API client 不可用',
    'errors.ollamaDiscoveryApiUnavailable': 'Ollama model discovery API client 不可用',
    'errors.localModelDiscoveryApiUnavailable': 'Local model discovery API client 不可用',
    'errors.settingsUpdateApiUnavailable': 'Settings update API client 不可用',
    'pathPicker.browseBackend': '瀏覽後端',
    'pathPicker.close': '關閉',
    'pathPicker.currentFolder': '使用目前資料夾',
    'pathPicker.description': '透過後端檔案系統清單瀏覽並選擇路徑',
    'pathPicker.directoryOnly': '僅顯示',
    'pathPicker.empty': '此路徑沒有可顯示項目',
    'pathPicker.errorList': '讀取路徑失敗',
    'pathPicker.errorRoots': '讀取根目錄失敗',
    'pathPicker.fileDisabledTitle': '目前模式僅可選擇資料夾',
    'pathPicker.itemNotSelectableTitle': '此項目不可選取',
    'pathPicker.loading': '載入中…',
    'pathPicker.loadingRoots': '正在讀取根目錄…',
    'pathPicker.noPath': '尚未選擇路徑',
    'pathPicker.parent': '上一層',
    'pathPicker.refresh': '重新整理',
    'pathPicker.selectDirectory': '選取資料夾',
    'pathPicker.selectFile': '選取檔案',
    'pathPicker.title': '選擇路徑',
    'settings.action.applyTest': '套用並測試連線',
    'settings.action.addApplyTest': '新增模型並測試連線',
    'settings.action.connect': '連線',
    'settings.action.importLocalModel': '匯入本機模型檔',
    'settings.action.saveLearning': '保存 Learning',
    'settings.action.saveMemory': '保存 Memory',
    'settings.action.saveVoice': '保存 Voice Pipeline',
    'settings.action.test': '測試',
    'settings.boolean.enabled': '已啟用',
    'settings.boolean.notEnabled': '未啟用',
    'settings.channel.allowedChannelIds': '允許頻道 ID',
    'settings.channel.allowedChatIds': '允許聊天室 ID',
    'settings.channel.allowedUserIds': '允許使用者 ID',
    'settings.channel.channelsRunner': 'Channels Runner',
    'settings.channel.commands': '常用指令',
    'settings.channel.enabledExternalChannels': '已啟用外部頻道',
    'settings.channel.enabledState': '啟用狀態',
    'settings.channel.localCli': '本機 CLI 通道',
    'settings.channel.registeredManager': '已註冊到 Manager',
    'settings.channel.running': '執行中',
    'settings.channel.token': 'Token',
    'settings.channel.discord.activeGuildId': '目前 Guild ID',
    'settings.channel.discord.activeSessionId': '目前 Session ID',
    'settings.channel.discord.activeVoiceChannelId': '目前語音頻道 ID',
    'settings.channel.discord.activeVoiceRooms': '作用中語音房',
    'settings.channel.discord.allowedGuildIds': '允許 Guild IDs',
    'settings.channel.discord.allowedVoiceChannelIds': '允許語音頻道 IDs',
    'settings.channel.discord.autoJoinPolicy': '自動加入策略',
    'settings.channel.discord.ingressGuildIds': 'Ingress Guild IDs',
    'settings.channel.discord.joinedAt': '加入時間',
    'settings.channel.discord.messageMode': '訊息模式',
    'settings.channel.discord.participants': '參與者',
    'settings.channel.discord.playback': '播放狀態',
    'settings.channel.discord.speaking': '說話狀態',
    'settings.channel.discord.textEnabled': '文字聊天已啟用',
    'settings.channel.discord.voiceEnabled': '語音已啟用',
    'settings.channel.discord.voiceError': '語音錯誤',
    'settings.channel.discord.voiceIngressEnabled': '語音收音已啟用',
    'settings.channel.discord.voiceIngressError': '語音收音錯誤',
    'settings.channel.discord.voiceReceiveExtension': '語音接收擴充',
    'settings.channel.discord.voiceRuntimePhase': '語音 Runtime 階段',
    'settings.disabled.connectUnavailable': '後端尚未提供 connect API',
    'settings.disabled.testUnavailable': '後端尚未提供 test API',
    'settings.navLabel': '設定導覽',
    'discordGuide.title': 'Discord 設定指南',
    'discordGuide.subtitle': '依照目前 Mochi 的產品化流程，完成 Discord bot 文字與語音接入。',
    'discordGuide.backToSettings': '返回設定',
    'discordGuide.navLabel': 'Discord 設定指南',
    'discordGuide.openPage': '開啟設定指南',
    'discordGuide.stepsTitle': '設定步驟',
    'discordGuide.powershellTitle': 'PowerShell',
    'discordGuide.configTitle': '最小設定範例',
    'discordGuide.footerNote': '若你已使用通道設定頁的 Discord 設定表單，token 不需要再手動出現在 settings API 回應中；本頁主要保留給需要理解底層配置流程的使用者。',
    'discordGuide.step1': '在 Discord Developer Portal 建立一個 Discord application 與 bot。',
    'discordGuide.step2': '在 bot 設定中啟用 Message Content Intent。',
    'discordGuide.step3': '使用 bot 與 applications.commands scopes 邀請 bot 進入你的伺服器。',
    'discordGuide.step4': '給 bot 權限：View Channels、Send Messages、Read Message History、Use Application Commands、Connect、Speak。',
    'discordGuide.step5': '在 Mochi 執行環境安裝 channels 相依套件。',
    'discordGuide.step6': '在 shell 或 process manager 中設定 DISCORD_BOT_TOKEN。',
    'discordGuide.step7': '在 WebGUI Discord 設定表單或本地 config 檔案中啟用 Discord。',
    'discordGuide.step8': '若使用獨立 channels runner，執行 uv run mochi channels run；若使用 WebGUI/API server，則可直接在 Channels 面板啟動 Discord。',
    'discordSetup.title': 'Discord 設定',
    'discordSetup.description': '使用專用安全流程接收 bot token，不需要手動編輯 config 檔案。',
    'discordSetup.botToken': 'Bot Token',
    'discordSetup.botTokenHelp': '此值只會送到專用的 Discord setup endpoint，不會由 settings API 回傳。',
    'discordSetup.botTokenPlaceholder': '貼上你的 Discord bot token',
    'discordSetup.enableChannel': '啟用 Discord 通道',
    'discordSetup.enableText': '啟用文字聊天',
    'discordSetup.enableVoice': '啟用語音房',
    'discordSetup.autoReply': '語音自動回覆',
    'discordSetup.voiceStt': '語音 STT 收音',
    'discordSetup.voiceTts': '語音 TTS 播放',
    'discordSetup.messageMode': '訊息模式',
    'discordSetup.messageMode.allMessages': '所有訊息',
    'discordSetup.messageMode.mentionsOnly': '只回應提及',
    'discordSetup.messageMode.slashOnly': '只回應斜線指令',
    'discordSetup.messageMode.all_messages': '所有訊息',
    'discordSetup.messageMode.mentions_only': '只回應提及',
    'discordSetup.messageMode.slash_only': '只回應斜線指令',
    'discordSetup.autoJoinPolicy': '自動加入策略',
    'discordSetup.autoJoinPolicy.manualOnly': '手動加入',
    'discordSetup.autoJoinPolicy.manual_only': '手動加入',
    'discordSetup.allowedGuildIds': '允許 Guild IDs',
    'discordSetup.allowedTextChannelIds': '允許文字頻道 IDs',
    'discordSetup.allowedVoiceChannelIds': '允許語音頻道 IDs',
    'discordSetup.allowedUserIds': '允許使用者 IDs',
    'discordSetup.idsPlaceholder': '1234567890, 2345678901',
    'discordSetup.save': '保存 Discord 設定',
    'discordSetup.successSaved': 'Discord 設定已保存；token 只會保存在本地，且不會出現在 API 回應中。',
    'discordSetup.errorSave': 'Discord 設定保存失敗',
    'discordSetup.errorApiUnavailable': 'Discord setup API client 不可用',
    'discordSetup.errorTokenRequired': '必須提供 Discord bot token',
    'settings.channel.discord.phase.idle': '待命',
    'settings.channel.discord.phase.connecting': '連線中',
    'settings.channel.discord.phase.ready': '就緒',
    'settings.channel.discord.phase.running': '執行中',
    'settings.channel.discord.phase.error': '錯誤',
    'settings.channel.discord.phase.closed': '已關閉',
    'settings.channel.discord.playbackState.idle': '待命',
    'settings.channel.discord.playbackState.playing': '播放中',
    'settings.channel.discord.playbackState.stopped': '已停止',
    'settings.channel.discord.playbackState.error': '錯誤',
    'settings.channel.discord.speakingState.idle': '待命',
    'settings.channel.discord.speakingState.speaking': '說話中',
    'settings.channel.discord.speakingState.silent': '靜音',
    'settings.channel.discord.speakingState.error': '錯誤',
    'channelControl.start': '啟動',
    'channelControl.stop': '停止',
    'channelControl.started': '{name} 已啟動。',
    'channelControl.stopped': '{name} 已停止。',
    'channelControl.errorStart': '無法啟動 {name}',
    'channelControl.errorStop': '無法停止 {name}',
    'channelControl.startApiUnavailable': 'Channels start API client 不可用',
    'channelControl.stopApiUnavailable': 'Channels stop API client 不可用',
    'settings.form.apiKey': 'API Key',
    'settings.form.apiUrl': 'API URL / Ollama Host',
    'settings.form.backend': 'Backend',
    'settings.form.device': '裝置',
    'settings.form.language': '語言',
    'settings.form.model': '模型',
    'settings.form.modelName': '模型名稱',
    'settings.form.speed': '速度',
    'settings.form.voice': 'Voice',
    'settings.learning.autoExtract': '自動萃取 Skill',
    'settings.learning.autoSyncFilesystem': '自動同步 SKILL.md',
    'settings.learning.description': '軌跡、技能庫、會話與保留策略',
    'settings.learning.enable': '啟用學習',
    'settings.learning.errorSave': '學習設定保存失敗',
    'settings.learning.improvementThreshold': '改進門檻',
    'settings.learning.maxSkills': '技能上限',
    'settings.learning.minSteps': '最少步數',
    'settings.learning.minToolCalls': '最少工具呼叫次數',
    'settings.learning.placeholder.pluginsDir': '例如 ~/.mochi/plugins',
    'settings.learning.placeholder.sessionsDir': '例如 ~/.mochi/sessions',
    'settings.learning.placeholder.skillsDir': '例如 ~/.mochi/skills',
    'settings.learning.placeholder.workspaceDir': '例如 ~/.mochi',
    'settings.learning.pluginsDir': 'Plugins 後端目錄',
    'settings.learning.retentionDays': '軌跡保留天數',
    'settings.learning.sessionsDir': 'Sessions 後端目錄',
    'settings.learning.skillsDir': 'Skills 後端目錄',
    'settings.learning.successSaved': '已保存學習與路徑設定',
    'settings.learning.title': 'Learning Storage',
    'settings.learning.workspaceDir': 'Workspace / Trajectories 後端目錄',
    'settings.memory.dbPath': '記憶 DB 後端路徑',
    'settings.memory.dbPathHelp': 'SQLite 檔案必須位於後端可存取的位置；Windows/macOS 瀏覽器不能直接提供可給後端 open 的絕對路徑。',
    'settings.memory.description': 'SQLite 記憶庫與檢索策略',
    'settings.memory.errorSave': '記憶設定保存失敗',
    'settings.memory.placeholder.dbPath': '例如 ~/.mochi/memory.db 或 /var/lib/mochi/memory.db',
    'settings.memory.shortTermMessages': '短期訊息數',
    'settings.memory.successSaved': '已保存記憶設定',
    'settings.memory.title': 'Memory Storage',
    'settings.modelConnection.apiKeyPlaceholderNoKey': 'Ollama 通常不需要 API key',
    'settings.modelConnection.description': '成功連線後會加入可用模型列表；可接 Ollama、OpenAI-compatible API，或掃描後端可存取的本地 GGUF / safetensors 權重路徑。API key 不會從設定 API 回傳。',
    'settings.modelConnection.errorConfigure': '模型設定失敗',
    'settings.modelConnection.ggufRuntimeMissing': 'GGUF 推理 runtime 尚未安裝。請先在設定頁的 `llama.cpp Runtime` 區塊安裝建議版本，或註冊既有路徑，再載入 GGUF 模型。',
    'settings.modelConnection.errorDiscover': '無法讀取 Ollama 模型',
    'settings.modelConnection.errorDiscoverLocal': '無法掃描本地模型',
    'settings.modelConnection.localModelPath': '模型權重路徑',
    'settings.modelConnection.localModelPlaceholder': '例如 /srv/mochi/models/qwen2.5.gguf 或 /srv/mochi/models/Qwen2.5-7B-Instruct',
    'settings.modelConnection.localRootPath': '本地模型根目錄',
    'settings.modelConnection.localRootPlaceholder': '例如 /srv/mochi/models 或 ~/.cache/huggingface/hub',
    'settings.modelConnection.refreshOllama': '刷新 Ollama 模型列表',
    'settings.modelConnection.scanLocal': '掃描本地模型',
    'settings.modelConnection.successDiscovered': '已讀取 {count} 個 Ollama 模型',
    'settings.modelConnection.successLocalDiscovered': '已找到 {count} 個本地模型',
    'settings.modelConnection.successNoLocalModels': '此路徑下沒有找到支援的本地模型',
    'settings.modelConnection.successNoModels': 'Ollama 可連線，但尚未回報模型',
    'settings.modelConnection.successSwitched': '已切換到 {model}',
    'settings.modelConnection.successSwitchedPersisted': '已切換並保存 {model}',
    'settings.modelConnection.title': '模型接入',
    'settings.quantization.title': '量化能力',
    'settings.quantization.description': '針對本地 Hugging Face 模型顯示 GGUF 轉換能力。GGUF 是目前 Mochi 的跨平台通用方案。',
    'settings.quantization.ggufHintTitle': '偵測到本地 Hugging Face / safetensors 權重',
    'settings.quantization.ggufHintBody': '目前選到的模型不是 GGUF。若要在 Mochi 的本地 llama.cpp runtime 取得較佳相容性，通常可轉換為 GGUF。',
    'settings.quantization.ggufHintDetail': '這個提示也適用於 FP8 等 safetensors 權重；它們雖然已量化，但在部分 GPU / runtime 上仍可能退回較高精度載入。',
    'settings.quantization.loading': '正在讀取量化能力…',
    'settings.quantization.errorLoad': '無法讀取量化能力',
    'settings.quantization.supportedLabel': '可用性',
    'settings.quantization.supportedValue': '可規劃',
    'settings.quantization.conditionalValue': '有條件可用',
    'settings.quantization.unsupportedValue': '尚不可用',
    'settings.quantization.reasonLabel': '說明',
    'settings.quantization.warningsLabel': '注意事項',
    'settings.quantization.optionsLabel': '可用量化選項',
    'settings.quantization.suggestedDefaultLabel': '系統建議預設',
    'settings.quantization.convertAction': '轉換為 GGUF',
    'settings.quantization.convertSuccess': 'GGUF 轉換完成',
    'settings.quantization.convertSuccessWithPath': 'GGUF 轉換完成：{path}',
    'settings.quantization.convertError': 'GGUF 轉換失敗',
    'settings.quantization.outputPathLabel': '輸出路徑',
    'settings.quantization.reasonMissing': '後端目前未回傳此格式的能力資料。',
    'settings.quantization.hardwareTitle': '硬體摘要',
    'settings.quantization.hardwareProvider': '探測來源',
    'settings.quantization.hardwareCuda': 'CUDA',
    'settings.quantization.hardwareGpuCount': 'GPU 數量',
    'settings.quantization.hardwareVram': 'VRAM',
    'settings.quantization.hardwarePrimaryGpu': '主要 GPU',
    'settings.quantization.hardwareWarningsLabel': '硬體探測警告',
    'settings.quantization.status.supported': 'Supported',
    'settings.quantization.status.unsupported': 'Unsupported',
    'settings.quantization.status.conditional': 'Conditional',
    'settings.quantization.status.unknown': 'Unknown',
    'settings.runtime.localRuntimeTitle': 'llama.cpp Runtime',
    'settings.runtime.localRuntimeDescription': '\u0047\u0047\u0055\u0046 \u6a21\u578b\u8f09\u5165\u8207 \u0047\u0047\u0055\u0046 \u8f49\u63db\u90fd\u4f9d\u8cf4\u53ef\u7528\u7684 llama.cpp runtime\u3002',
    'settings.runtime.localRuntimeLoading': '\u6b63\u5728\u8b80\u53d6 llama.cpp runtime \u72c0\u614b...',
    'settings.runtime.localRuntimeErrorLoad': '\u7121\u6cd5\u8b80\u53d6 llama.cpp runtime \u72c0\u614b',
    'settings.runtime.localRuntimeInstallAction': '\u5b89\u88dd\u5efa\u8b70\u7248\u672c',
    'settings.runtime.localRuntimeRegisterAction': '\u4f7f\u7528\u73fe\u6709\u8def\u5f91',
    'settings.runtime.localRuntimeInstallPrepared': '\u5df2\u6e96\u5099 managed runtime \u76ee\u9304\u3002',
    'settings.runtime.localRuntimeInstallError': '\u7121\u6cd5\u6e96\u5099 llama.cpp runtime',
    'settings.runtime.localRuntimeRegistered': '\u5df2\u8a3b\u518a\u73fe\u6709 llama.cpp \u8def\u5f91\u3002',
    'settings.runtime.localRuntimeRegisterError': '\u7121\u6cd5\u8a3b\u518a\u73fe\u6709 llama.cpp \u8def\u5f91',
    'settings.runtime.localRuntimeExistingPath': '\u65e2\u6709 llama.cpp \u8def\u5f91',
    'settings.runtime.localRuntimeExistingPathPlaceholder': '\u4f8b\u5982 /opt/llama.cpp \u6216 /workspace/llama.cpp',
    'settings.runtime.localRuntimeExistingPathRequired': '\u8acb\u5148\u8f38\u5165\u65e2\u6709 llama.cpp \u8def\u5f91',
    'settings.runtime.localRuntimeSource': '\u4f86\u6e90',
    'settings.runtime.localRuntimeVersion': '\u7248\u672c',
    'settings.runtime.localRuntimeRoot': '\u6839\u76ee\u9304',
    'settings.runtime.localRuntimeMissing': '\u7f3a\u5c11\u5143\u4ef6',
    'settings.runtime.localRuntimeBlocked': '\u8acb\u5148\u6e96\u5099 llama.cpp runtime',
    'settings.runtime.localRuntimeState.ready': 'Ready',
    'settings.runtime.localRuntimeState.degraded': 'Degraded',
    'settings.runtime.localRuntimeState.missing': 'Missing',
    'settings.runtime.localRuntimeState.not_installed': 'Not installed',
    'settings.runtime.localRuntimeState.manual_setup_required': 'Manual setup',
    'settings.runtime.localRuntimeState.incompatible': 'Incompatible',
    'settings.runtime.localRuntimeState.installing': 'Installing',
    'settings.runtime.localRuntimeState.unknown': 'Unknown',
    'settings.preferences.timezone.autoWithZone': '跟隨瀏覽器（{timezone}）',
    'settings.provider.anthropic.description': 'Claude OpenAI SDK compatibility endpoint',
    'settings.provider.anthropic.note': '目前走 Anthropic OpenAI 相容層；原生 Anthropic API 尚未實作。',
    'settings.provider.gemini.description': 'Google Gemini OpenAI-compatible endpoint',
    'settings.provider.gemini.note': '目前走 Gemini 官方 OpenAI-compatible endpoint。',
    'settings.provider.local.description': '直接載入後端本地 GGUF 或 Hugging Face safetensors 權重',
    'settings.provider.local.note': '路徑必須是 Mochi 後端進程可讀取的 server-side path；MVP 會在 API 進程內載入模型，較大模型可能需要較多 RAM/VRAM。',
    'settings.provider.ollama.description': '本機 Ollama HTTP API',
    'settings.provider.ollama.note': '連上後會讀取 /api/tags 並使用模型下拉選單。',
    'settings.provider.openaiCompat.description': 'OpenAI-compatible Chat Completions 或 Responses endpoint',
    'settings.provider.openaiCompat.note': '若只填到 /v1 會使用 /chat/completions；若填完整 /responses 或 /chat/completions 則原樣使用。',
    'settings.provider.openaiCodex.description': '使用 ChatGPT OAuth 的 OpenAI Codex backend-api transport',
    'settings.provider.openaiCodex.note': '此路徑不使用 API key。請先從本機 Codex CLI 匯入 ChatGPT 登入，再連接模型。',
    'settings.voice.description': 'STT/TTS runtime 與本機模型保存位置',
    'settings.voice.downloadMissing': '缺少模型時準備下載',
    'settings.voice.enable': '啟用語音',
    'settings.voice.enableHelp': '保存後新的 voice session 會使用更新後設定。',
    'settings.voice.errorImport': '模型匯入失敗',
    'settings.voice.errorSave': '語音設定保存失敗',
    'settings.voice.importSuccess': '已匯入 {count} 個檔案（{bytes}），並填入後端路徑。請保存 Voice Pipeline 讓設定生效。',
    'settings.voice.placeholder.sttCacheDir': '例如 ~/.cache/mochi/stt 或 /var/lib/mochi/models/stt',
    'settings.voice.placeholder.sttModelPath': '例如 /srv/mochi/models/whisper/large-v3 或單一模型檔',
    'settings.voice.saveSuccess': '已保存語音設定',
    'settings.voice.saveSuccessWithStatus': '已保存語音設定；模型狀態：{status}',
    'settings.voice.sttCacheDir': 'STT 後端模型快取/下載目錄',
    'settings.voice.sttCacheDirHelp': '後端實際讀寫的伺服器路徑。WSL 後端會使用 Linux/WSL 路徑，不是瀏覽器本機路徑。',
    'settings.voice.sttModelPath': 'STT 後端本地模型路徑（可選覆寫）',
    'settings.voice.sttModelPathHelp': '手動輸入後端既有模型檔案或資料夾路徑；大型資料夾不再透過瀏覽器枚舉，避免瀏覽器崩潰。',
    'settings.voice.sttTitle': 'Speech to Text',
    'settings.voice.ttsTitle': 'Text to Speech',
    'settings.voice.title': 'Voice Pipeline',
    'sidebar.collapse': '收合側欄',
    'sidebar.deleteConversation': '刪除對話',
    'sidebar.deleteConfirm': '刪除「{title}」？',
    'sidebar.deleteDialogAction': '刪除',
    'sidebar.deleteDialogDescription': '這會永久刪除「{title}」這個對話。',
    'sidebar.deleteDialogTitle': '刪除對話？',
    'sidebar.bulkDelete': '批量刪除',
    'sidebar.bulkCancel': '取消選取',
    'sidebar.bulkSelectAll': '全選可見項目',
    'sidebar.bulkClearAll': '清除全選',
    'sidebar.bulkDeleteSelected': '刪除已選',
    'sidebar.bulkSelectedCount': '已選 {count} 項',
    'sidebar.browseFolder': '瀏覽資料夾',
    'sidebar.bulkDeleteDialogTitle': '刪除已選對話？',
    'sidebar.bulkDeleteDialogDescription': '這會永久刪除 {count} 個對話。',
    'sidebar.expand': '展開側欄',
    'sidebar.newChat': '新對話',
    'sidebar.newChatShortcut': '新對話 (⌘/)',
    'sidebar.noResults': '無符合搜尋結果',
    'sidebar.noSessions': '尚無對話記錄',
    'sidebar.older': '更早',
    'sidebar.pinned': '置頂',
    'sidebar.projectDirectoryPickerFailed': '無法開啟資料夾選擇器',
    'sidebar.rename': '重新命名',
    'sidebar.renameCancel': '取消改名',
    'sidebar.renameSave': '保存名稱',
    'sidebar.searchPlaceholder': '搜尋對話… (⌘K)',
    'app.shortcuts.openInput': '聚焦輸入框 (⌘L)',
    'app.shortcuts.toggleSidebar': '切換側欄 (⌘B)',
    'sidebar.settings': '設定',
    'sidebar.skills': 'Skill 庫',
    'sidebar.workflows': '工作流',
    'sidebar.goals': '目標',
    'sidebar.thisWeek': '本週',
    'sidebar.today': '今天',
    'workflows.apiUnavailable': '這個後端目前尚未提供 Workflows API。',
    'workflows.backToList': '返回工作流',
    'workflows.create': '建立工作流',
    'workflows.createDescription': '設定 protocol 與 subagent。模型來自已設定的 providers。',
    'workflows.createError': '無法建立工作流。',
    'workflows.description': '建立並監看結構化多代理工作流，支援協作、蒐證、評分與排程。',
    'workflows.loadError': '無法載入工作流。',
    'workflows.title': '工作流',
    'agentRuns.status.created': '已建立',
    'agentRuns.status.running': '執行中',
    'agentRuns.status.queued': '排隊中',
    'agentRuns.status.pending': '等待中',
    'agentRuns.status.paused': '已暫停',
    'agentRuns.status.awaiting_resources': '等待資源',
    'agentRuns.status.stalled': '卡住',
    'agentRuns.status.partial': '部分完成',
    'agentRuns.status.degraded': '降級',
    'agentRuns.status.succeeded': '成功',
    'agentRuns.status.completed': '已完成',
    'agentRuns.status.done': '已完成',
    'agentRuns.status.failed': '失敗',
    'agentRuns.status.cancelled': '已取消',
    'agentRuns.status.error': '錯誤',
    'agentRuns.badge.degraded': '降級',
    'agentRuns.badge.finalizePartialReady': '可保留 partial 結束',
    'agentRuns.runPolicy.title': 'Run Policy',
    'agentRuns.runPolicy.description': '限制總執行時間、偵測 subagent 卡住，並定義資源不足或角色掉線時的恢復策略。',
    'agentRuns.runPolicy.preset': '預設組合',
    'agentRuns.runPolicy.presetPlaceholder': '選擇 run policy 預設',
    'agentRuns.runPolicy.short': '短任務',
    'agentRuns.runPolicy.balanced': '平衡',
    'agentRuns.runPolicy.long': '長研究',
    'agentRuns.runPolicy.custom': '自訂',
    'agentRuns.runPolicy.maxWallClock': '最長執行時間（秒）',
    'agentRuns.runPolicy.heartbeatTimeout': 'Heartbeat 逾時（秒）',
    'agentRuns.runPolicy.checkpointInterval': 'Checkpoint 間隔',
    'agentRuns.runPolicy.maxFailuresPerRole': '每角色最大失敗數',
    'agentRuns.runPolicy.budgetExhausted': '預算耗盡時',
    'agentRuns.runPolicy.budgetPlaceholder': '選擇預算耗盡動作',
    'agentRuns.runPolicy.disconnect': 'Subagent 斷線時',
    'agentRuns.runPolicy.disconnectPlaceholder': '選擇斷線恢復動作',
    'agentRuns.runPolicy.budgetPause': '轉為等待資源',
    'agentRuns.runPolicy.budgetFinalizePartial': '保留 partial 並結束',
    'agentRuns.runPolicy.disconnectRetryThenDegrade': '先重試，再降級',
    'agentRuns.runPolicy.disconnectPause': '轉為 stalled',
    'agentRuns.runPolicy.disconnectFail': '直接失敗',
    'agentRuns.recentRuns.title': '最近 Runs',
    'agentRuns.recentRuns.description': '開啟 run 查看事件、artifact、recovery 提示與 guidance 操作。',
    'agentRuns.recentRuns.empty': '目前還沒有 agent runs。',
    'agentRuns.run.untitled': '未命名 run',
    'agentRuns.run.noTopic': '未提供主題。',
    'agentRuns.run.updated': '更新時間：{value}',
    'agentRuns.run.state.active': '進行中',
    'agentRuns.run.state.finished': '已結束',
    'agentRuns.recovery.title': 'Recovery',
    'agentRuns.recovery.description': '目前 recovery state、有效 run policy，以及 subagent health 摘要。',
    'agentRuns.recovery.operatorTitle': 'Operator 提示',
    'agentRuns.recovery.resumeTitle': 'Resume 提示',
    'agentRuns.recovery.policyTitle': 'Run Policy',
    'agentRuns.recovery.stateTitle': 'Recovery State',
    'agentRuns.recovery.healthTitle': 'Subagent Health',
    'agentRuns.recovery.emptyHealth': '目前沒有回報額外的 subagent health snapshot。',
    'agentRuns.recovery.awaiting_resources.message': '這個 run 正在等待額外資源。補齊模型、工具權限、資料或預算後再 resume。',
    'agentRuns.recovery.awaiting_resources.resume': '確認缺的資源已補齊，再按 Resume 繼續目前 attempt。',
    'agentRuns.recovery.stalled.message': '這個 run 目前卡住。先檢查 subagent health、最新錯誤與 detached exec 狀態，再決定是否 resume。',
    'agentRuns.recovery.stalled.resume': '若外部程序或缺失依賴已恢復，可直接 Resume；若仍異常，先送出 guidance 或停止背景 job。',
    'agentRuns.recovery.partial.message': '這個 run 已保留部分輸出。你可以先檢查現有 artifacts，再決定是否補充指示後 resume。',
    'agentRuns.recovery.partial.resume': 'Resume 會沿用目前上下文與既有成果，適合在確認方向後繼續補完。',
    'agentRuns.recovery.degraded.message': '這個 run 已進入降級模式，代表部分角色或能力不可用，但流程仍可在較弱配置下繼續。',
    'agentRuns.recovery.degraded.resume': '若要繼續，先確認降級是否可接受；若不接受，先修復依賴或更換模型後再 resume。',
    'agentRuns.recovery.default.message': '目前沒有額外 recovery 指示，可直接查看事件與 artifacts 判斷下一步。',
    'agentRuns.recovery.default.resume': '若 run 已暫停且條件已滿足，可以直接 Resume。',
    'agentRuns.recovery.operatorFallback': '後端尚未提供 operator message，以下為前端依狀態推導的操作建議。',
    'agentRuns.recovery.latestError': '最新錯誤',
    'agentRuns.recovery.suggestedAction': '建議動作',
    'agentRuns.recovery.finalizePartialAvailable': '目前狀態可直接保留為 partial 結束，避免繼續卡在等待資源或 stalled 階段。',
    'agentRuns.recovery.finalizePartialReason': 'Finalize partial 說明',
    'agentRuns.actions.finalizePartial': '保留為 Partial',
    'agentRuns.exec.title': 'Detached Exec Jobs',
    'agentRuns.exec.description': '查看背景 exec job 狀態、重新接回輸出，或要求停止目前 session。',
    'agentRuns.exec.empty': '目前沒有 detached exec jobs。',
    'agentRuns.exec.emptyStalled': '目前沒有回報 detached exec jobs；如果 run 仍 stalled，優先檢查 subagent health、最新錯誤與外部依賴。',
    'agentRuns.exec.command': '命令',
    'agentRuns.exec.status': '狀態',
    'agentRuns.exec.liveStatus': '即時狀態',
    'agentRuns.exec.leaseOwner': '持有者',
    'agentRuns.exec.workdir': '工作目錄',
    'agentRuns.exec.logPath': '日誌路徑',
    'agentRuns.exec.approval': '審批狀態',
    'agentRuns.exec.background': '背景執行',
    'agentRuns.exec.timeout': '逾時',
    'agentRuns.exec.pid': 'PID',
    'agentRuns.exec.reattachSupported': '可重新接回',
    'agentRuns.exec.stopStatus': '停止結果',
    'agentRuns.exec.reattachStatus': '接回結果',
    'agentRuns.exec.noCommand': '未提供命令內容。',
    'agentRuns.exec.noSnapshot': '尚未抓取這個 session 的輸出快照。',
    'agentRuns.exec.stdout': '標準輸出',
    'agentRuns.exec.stderr': '標準錯誤',
    'agentRuns.exec.stdoutEmpty': '目前沒有 stdout。',
    'agentRuns.exec.stderrEmpty': '目前沒有 stderr。',
    'agentRuns.exec.snapshotTitle': '最新 Session Snapshot',
    'agentRuns.exec.snapshotDescription': '這個快照來自最近一次 poll / reattach / stop 操作。',
    'agentRuns.exec.rawPayload': '原始 payload',
    'agentRuns.exec.action.poll': '更新狀態',
    'agentRuns.exec.action.reattach': '重新接回',
    'agentRuns.exec.action.stop': '停止',
    'agentRuns.exec.action.unavailable': '不可用',
    'agentRuns.exec.boolean.yes': '是',
    'agentRuns.exec.boolean.no': '否',
    'agentRuns.exec.status.running': '執行中',
    'agentRuns.exec.status.pending': '等待中',
    'agentRuns.exec.status.queued': '排隊中',
    'agentRuns.exec.status.completed': '已完成',
    'agentRuns.exec.status.succeeded': '成功',
    'agentRuns.exec.status.failed': '失敗',
    'agentRuns.exec.status.error': '錯誤',
    'agentRuns.exec.status.stopped': '已停止',
    'agentRuns.exec.status.unavailable': '不可用',
    'skills.card.created': 'Created',
    'skills.card.delete': '刪除 {name}',
    'skills.card.noDescription': '無描述',
    'skills.card.success': 'Success',
    'skills.card.unknown': 'Unknown',
    'skills.card.untagged': 'untagged',
    'skills.card.used': 'Used',
    'skills.empty': '目前沒有可顯示的 Skill',
    'skills.errorDelete': '刪除 Skill 失敗',
    'skills.errorLoad': '載入 Skill 失敗',
    'skills.loaded': 'Loaded',
    'skills.noSearchResults': '找不到符合「{query}」的 Skill',
    'skills.refresh': '重新整理',
    'skills.searchPlaceholder': '搜尋名稱、描述或標籤',
    'skills.subtitle': '後端 Skill 索引與使用統計摘要',
    'skills.tabs.all': '全部',
    'skills.tabs.highestSuccess': '高成功率',
    'skills.tabs.mostUsed': '常用',
    'skills.tabs.recent': '最新',
    'skills.title': 'Skill 庫',
    'settings.title': '設定',
    'settings.subtitle': '後端設定摘要與服務可用性',
    'settings.errorLoadFailed': '載入設定失敗',
    'settings.badge.connected': 'Connected',
    'settings.badge.partial': 'Partial',
    'settings.stats.primaryModel': '主要模型',
    'settings.stats.modelEndpoints': '模型端點',
    'settings.stats.channels': '通道數',
    'settings.stats.webSurface': 'Web 介面',
    'settings.stats.configured': '已設定',
    'settings.stats.notReported': '未回報',
    'settings.tabs.model': '模型',
    'settings.tabs.inference': '推理',
    'settings.tabs.voice': '語音',
    'settings.tabs.memory': '記憶',
    'settings.tabs.learning': '學習',
    'settings.tabs.security': '安全',
    'settings.tabs.channels': '通道',
    'settings.tabs.web': 'Web',
    'settings.action.saveSecurity': '儲存安全設定',
    'settings.security.title': '自主與安全',
    'settings.security.description': '調整執行自主度，以及 exec / 檔案工具的預設審批策略。',
    'settings.security.autonomyMode': '自主模式',
    'settings.security.autonomyMode.strict': 'Strict',
    'settings.security.autonomyMode.trusted_workspace': 'Trusted workspace',
    'settings.security.autonomyMode.auto_review': '自動審查',
    'settings.security.autonomyMode.high_autonomy': '高自主',
    'settings.security.requireFileWriteApproval': '檔案寫入需審批',
    'settings.security.requireExecApproval': 'Exec 執行需審批',
    'settings.security.execDefaultTimeoutSec': 'Exec 預設逾時（秒）',
    'settings.security.execSessionOutputLimit': 'Exec 輸出尾端上限（字元）',
    'settings.security.agentRunDefaultMaxWallClockSec': 'Agent Run 預設最長執行時間（秒）',
    'settings.security.agentRunDefaultMaxWallClockSecPlaceholder': '留空表示不設預設上限',
    'settings.security.agentRunDefaultOnBudgetExhausted': 'Agent Run 預設超時動作',
    'settings.security.agentRunDefaultOnBudgetExhausted.pause': '轉為 awaiting_resources',
    'settings.security.agentRunDefaultOnBudgetExhausted.finalize_partial': '結束並保留 partial',
    'settings.security.agentRunDefaultHeartbeatTimeoutSec': 'Agent Run 預設 heartbeat timeout (秒)',
    'settings.security.agentRunDefaultHeartbeatTimeoutSecPlaceholder': '留白表示不啟用 stalled watchdog',
    'settings.security.agentRunDefaultCheckpointIntervalSteps': 'Agent Run 預設 checkpoint 間隔步數',
    'settings.security.agentRunDefaultMaxSubagentFailuresPerRole': '每個 subagent role 預設最大失敗次數',
    'settings.security.agentRunDefaultOnSubagentDisconnect': 'Subagent 中斷時預設動作',
    'settings.security.agentRunDefaultOnSubagentDisconnect.retry_then_degrade': '重試後降級繼續',
    'settings.security.agentRunDefaultOnSubagentDisconnect.pause': '轉為 stalled',
    'settings.security.agentRunDefaultOnSubagentDisconnect.fail': '直接失敗',
    'settings.security.fileScope': '檔案操作範圍',
    'settings.security.fileScope.workspace': '僅 workspace',
    'settings.security.fileScope.any': '任意路徑 (較高風險)',
    'settings.security.maxWriteSizeMb': '單次寫入大小上限 (MB)',
    'settings.security.undoMaxSizeMb': 'Undo 最大內容大小 (MB)',
    'settings.security.successSaved': '安全設定已更新',
    'settings.security.errorSave': '安全設定更新失敗',
    'settings.section.modelConfig': '模型設定',
    'settings.section.discoveredModels': '已發現模型',
    'settings.savedModels.title': '已保存模型',
    'settings.savedModels.edit': '編輯',
    'settings.savedModels.delete': '刪除',
    'settings.savedModels.save': '保存',
    'settings.savedModels.cancel': '取消',
    'settings.savedModels.apiKeyHint': '留空則保留現有 API key；填入新值會覆蓋。',
    'settings.savedModels.none': '目前沒有已保存模型',
    'settings.savedModels.errorUpdate': '保存模型項目失敗',
    'settings.savedModels.errorDelete': '刪除模型項目失敗',
    'settings.savedModels.successUpdate': '模型項目已更新',
    'settings.savedModels.successDelete': '模型項目已刪除',
    'settings.section.voicePipeline': '語音管線',
    'settings.section.runtime': '執行狀態',
    'settings.section.memory': '記憶',
    'settings.section.learning': '學習',
    'settings.section.channels': '通道',
    'settings.section.web': 'Web',
    'settings.summary.modelConfig': '只顯示非敏感後端欄位',
    'settings.summary.discoveredModels': '來自 models API 的可用模型摘要',
    'settings.summary.voicePipeline': 'STT、TTS 與輸入輸出裝置摘要',
    'settings.summary.memory': '儲存後端、路徑與保留策略摘要',
    'settings.summary.learning': 'Skill 萃取與軌跡保留摘要',
    'settings.summary.channels': '只顯示通道狀態，不暴露 bot token 或 API key',
    'settings.summary.web': 'Web 服務與前端對接設定摘要',
    'settings.runtime.voiceFields': '語音欄位',
    'settings.runtime.settingsSource': '設定來源',
    'settings.runtime.backend': '後端',
    'settings.runtime.unavailable': '不可用',
    'settings.preferences.title': '介面偏好',
    'settings.preferences.description': '調整語言、外觀、整體字級與時區，偏好只會儲存在目前瀏覽器。',
    'settings.preferences.language': '介面語言',
    'settings.preferences.appearance': '外觀模式',
    'settings.preferences.fontSize': '字級',
    'settings.preferences.timezone': '時區',
    'settings.preferences.appearanceHelp': '外觀可跟隨系統，或手動固定為深色/淺色。',
    'settings.preferences.fontHelp': '字級會影響整個 WebGUI 主要介面。',
    'settings.preferences.timezoneHelp': '時區只影響顯示層；資料儲存與 API 仍維持 UTC。',
    'settings.preferences.language.default': '預設 / 自動偵測',
    'settings.preferences.language.zhTW': '繁體中文',
    'settings.preferences.language.en': 'English',
    'settings.preferences.appearance.system': '系統',
    'settings.preferences.appearance.dark': '深色',
    'settings.preferences.appearance.light': '淺色',
    'settings.preferences.font.compact': '緊湊',
    'settings.preferences.font.default': '預設',
    'settings.preferences.font.large': '較大',
    'settings.preferences.timezone.auto': '跟隨瀏覽器',
    'settings.preferences.timezone.utc': 'UTC',
    'settings.emptyState': '後端尚未回報可顯示的設定摘要。',
    'settings.noSummary': '目前沒有可顯示摘要',
    'settings.modelFallback': '模型',
  },
  en: {
    'common.available': 'Available',
    'common.cancel': 'Cancel',
    'common.close': 'Close',
    'common.configured': 'Configured',
    'common.disabled': 'Disabled',
    'common.enabled': 'Enabled',
    'common.fields': '{count} fields',
    'common.items': '{count} items',
    'common.listSummary': '{items} and {count} total',
    'common.loading': 'Loading...',
    'common.none': 'None',
    'common.no': 'No',
    'common.notConfigured': 'Not configured',
    'common.notReported': 'Not reported',
    'common.notSet': 'Not set',
    'common.refresh': 'Refresh',
    'common.reported': 'Reported',
    'common.unavailable': 'Unavailable',
    'common.unknown': 'Unknown',
    'common.yes': 'Yes',
    'chat.disclaimer': 'Mochi can make mistakes. Verify important information.',
    'chat.emptyAssistantResponse': 'The agent did not return any content.',
    'chat.input.attachFile': 'Upload file',
    'chat.input.currentModel': 'Current model',
    'chat.input.placeholder': 'Message Mochi... (Enter to send, Shift+Enter for newline)',
    'chat.input.send': 'Send (Enter)',
    'chat.input.voice': 'Voice input (⌘⇧V)',
    'chat.loadingLocalModel': 'Loading local model. This may take a while...',
    'chat.loadingSession': 'Loading conversation history...',
    'chat.moreOptions': 'More options',
    'chat.newChat': 'New chat',
    'chat.modelSwitchFailed': 'Model switch failed',
    'chat.requestFailed': 'Unable to complete this chat request',
    'chat.settingsShortcut': 'Settings (⌘,)',
    'chat.system.ready': 'Mochi WebGUI is ready. Messages will show reasoning, tool calls, and final answer events.',
    'chat.thinking': 'Thinking',
    'chat.voice.assistant': 'Assistant',
    'chat.voice.close': 'Close',
    'chat.voice.deviceUnknown': 'Unknown microphone',
    'chat.voice.interrupt': 'Interrupt',
    'chat.voice.microphone': 'Microphone',
    'chat.voice.noInputDetected': 'No microphone input signal detected yet. Check browser permission and the selected recording device.',
    'chat.voice.phase.connecting': 'Connecting',
    'chat.voice.phase.error': 'Error',
    'chat.voice.phase.idle': 'Idle',
    'chat.voice.phase.listening': 'Listening',
    'chat.voice.phase.ready': 'Ready',
    'chat.voice.phase.synthesizing': 'Synthesizing',
    'chat.voice.phase.thinking': 'Thinking',
    'chat.voice.phase.transcribing': 'Transcribing',
    'chat.voice.record': 'Record',
    'chat.voice.signalDetected': 'Microphone signal detected',
    'chat.voice.signalWaiting': 'Waiting for microphone signal',
    'chat.voice.startPrompt': 'Start recording to speak.',
    'chat.voice.stop': 'Stop',
    'chat.voice.title': 'Voice Session',
    'chat.voice.transcription': 'Transcription',
    'chat.voice.vadDetected': 'Speech detected',
    'chat.voice.waiting': 'Waiting for response.',
    'chat.tool.args': 'Arguments',
    'chat.tool.error': 'Error',
    'chat.tool.failed': 'Failed',
    'chat.tool.result': 'Result',
    'chat.tool.running': 'Running...',
    'errors.channelsStatusUnavailable': 'The backend does not expose the /v1/channels status endpoint',
    'errors.chatApiUnavailable': 'Chat API client is unavailable',
    'errors.filesystemImportUnavailable': 'Filesystem import API client is unavailable',
    'errors.modelConfigureApiUnavailable': 'Model configure API client is unavailable',
    'errors.ollamaDiscoveryApiUnavailable': 'Ollama model discovery API client is unavailable',
    'errors.localModelDiscoveryApiUnavailable': 'Local model discovery API client is unavailable',
    'errors.settingsUpdateApiUnavailable': 'Settings update API client is unavailable',
    'pathPicker.browseBackend': 'Browse backend',
    'pathPicker.close': 'Close',
    'pathPicker.currentFolder': 'Use current folder',
    'pathPicker.description': 'Browse the backend filesystem and choose a path',
    'pathPicker.directoryOnly': 'Directory only',
    'pathPicker.empty': 'No displayable items in this path',
    'pathPicker.errorList': 'Failed to read path',
    'pathPicker.errorRoots': 'Failed to read root directories',
    'pathPicker.fileDisabledTitle': 'Current mode only allows directories',
    'pathPicker.itemNotSelectableTitle': 'This item cannot be selected',
    'pathPicker.loading': 'Loading...',
    'pathPicker.loadingRoots': 'Reading root directories...',
    'pathPicker.noPath': 'No path selected',
    'pathPicker.parent': 'Parent',
    'pathPicker.refresh': 'Refresh',
    'pathPicker.selectDirectory': 'Select directory',
    'pathPicker.selectFile': 'Select file',
    'pathPicker.title': 'Choose path',
    'settings.action.applyTest': 'Apply and test connection',
    'settings.action.addApplyTest': 'Add model and test connection',
    'settings.action.addModel': 'Add Model',
    'settings.action.connect': 'Connect',
    'settings.action.importLocalModel': 'Import local model file',
    'settings.action.saveLearning': 'Save Learning',
    'settings.action.saveMemory': 'Save Memory',
    'settings.action.saveVoice': 'Save Voice Pipeline',
    'settings.action.test': 'Test',
    'settings.action.testConnection': 'Test Connection',
    'settings.boolean.enabled': 'Enabled',
    'settings.boolean.notEnabled': 'Disabled',
    'settings.channel.allowedChannelIds': 'Allowed channel IDs',
    'settings.channel.allowedChatIds': 'Allowed chat IDs',
    'settings.channel.allowedUserIds': 'Allowed user IDs',
    'settings.channel.channelsRunner': 'Channels Runner',
    'settings.channel.commands': 'Common commands',
    'settings.channel.enabledExternalChannels': 'Enabled external channels',
    'settings.channel.enabledState': 'Enabled',
    'settings.channel.localCli': 'Local CLI channel',
    'settings.channel.registeredManager': 'Registered with Manager',
    'settings.channel.running': 'Running',
    'settings.channel.token': 'Token',
    'settings.channel.discord.activeGuildId': 'Active Guild ID',
    'settings.channel.discord.activeSessionId': 'Active Session ID',
    'settings.channel.discord.activeVoiceChannelId': 'Active Voice Channel ID',
    'settings.channel.discord.activeVoiceRooms': 'Active Voice Rooms',
    'settings.channel.discord.allowedGuildIds': 'Allowed Guild IDs',
    'settings.channel.discord.allowedVoiceChannelIds': 'Allowed Voice Channel IDs',
    'settings.channel.discord.autoJoinPolicy': 'Auto Join Policy',
    'settings.channel.discord.ingressGuildIds': 'Ingress Guild IDs',
    'settings.channel.discord.joinedAt': 'Joined At',
    'settings.channel.discord.messageMode': 'Message Mode',
    'settings.channel.discord.participants': 'Participants',
    'settings.channel.discord.playback': 'Playback',
    'settings.channel.discord.speaking': 'Speaking',
    'settings.channel.discord.textEnabled': 'Text Enabled',
    'settings.channel.discord.voiceEnabled': 'Voice Enabled',
    'settings.channel.discord.voiceError': 'Voice Error',
    'settings.channel.discord.voiceIngressEnabled': 'Voice Ingress Enabled',
    'settings.channel.discord.voiceIngressError': 'Voice Ingress Error',
    'settings.channel.discord.voiceReceiveExtension': 'Voice Receive Extension',
    'settings.channel.discord.voiceRuntimePhase': 'Voice Runtime Phase',
    'settings.disabled.connectUnavailable': 'The backend does not provide a connect API yet',
    'settings.disabled.testUnavailable': 'The backend does not provide a test API yet',
    'settings.navLabel': 'Settings navigation',
    'discordGuide.title': 'Discord Setup Guide',
    'discordGuide.subtitle': 'Follow the current Mochi product flow to connect Discord bot text and voice.',
    'discordGuide.backToSettings': 'Back to settings',
    'discordGuide.navLabel': 'Discord Setup Guide',
    'discordGuide.openPage': 'Open setup guide',
    'discordGuide.stepsTitle': 'Setup steps',
    'discordGuide.powershellTitle': 'PowerShell',
    'discordGuide.configTitle': 'Minimal config',
    'discordGuide.footerNote': 'If you already use the Discord Setup form above, the token no longer needs to appear anywhere in settings API responses. This page mainly remains for users who want to understand the underlying configuration flow.',
    'discordGuide.step1': 'Create a Discord application and bot in the Discord Developer Portal.',
    'discordGuide.step2': 'Enable the Message Content Intent in the bot settings.',
    'discordGuide.step3': 'Invite the bot with the bot and applications.commands scopes.',
    'discordGuide.step4': 'Grant permissions: View Channels, Send Messages, Read Message History, Use Application Commands, Connect, Speak.',
    'discordGuide.step5': 'Install channel dependencies in your Mochi environment.',
    'discordGuide.step6': 'Set DISCORD_BOT_TOKEN in your shell or process manager.',
    'discordGuide.step7': 'Enable Discord in the WebGUI Discord Setup form or in a local config file.',
    'discordGuide.step8': 'If you use the standalone channels runner, run uv run mochi channels run; if you use the WebGUI/API server, you can start Discord directly from the Channels panel.',
    'discordSetup.title': 'Discord Setup',
    'discordSetup.description': 'Use the dedicated secure flow for bot token onboarding instead of editing config files by hand.',
    'discordSetup.botToken': 'Bot Token',
    'discordSetup.botTokenHelp': 'This value is sent only to the dedicated Discord setup endpoint and is never returned by the settings API.',
    'discordSetup.botTokenPlaceholder': 'Paste your Discord bot token',
    'discordSetup.enableChannel': 'Enable Discord channel',
    'discordSetup.enableText': 'Enable text chat',
    'discordSetup.enableVoice': 'Enable voice rooms',
    'discordSetup.autoReply': 'Auto-reply in voice',
    'discordSetup.voiceStt': 'Voice STT ingest',
    'discordSetup.voiceTts': 'Voice TTS playback',
    'discordSetup.messageMode': 'Message Mode',
    'discordSetup.messageMode.allMessages': 'All messages',
    'discordSetup.messageMode.mentionsOnly': 'Mentions only',
    'discordSetup.messageMode.slashOnly': 'Slash commands only',
    'discordSetup.messageMode.all_messages': 'All messages',
    'discordSetup.messageMode.mentions_only': 'Mentions only',
    'discordSetup.messageMode.slash_only': 'Slash commands only',
    'discordSetup.autoJoinPolicy': 'Auto Join Policy',
    'discordSetup.autoJoinPolicy.manualOnly': 'Manual only',
    'discordSetup.autoJoinPolicy.manual_only': 'Manual only',
    'discordSetup.allowedGuildIds': 'Allowed Guild IDs',
    'discordSetup.allowedTextChannelIds': 'Allowed Text Channel IDs',
    'discordSetup.allowedVoiceChannelIds': 'Allowed Voice Channel IDs',
    'discordSetup.allowedUserIds': 'Allowed User IDs',
    'discordSetup.idsPlaceholder': '1234567890, 2345678901',
    'discordSetup.save': 'Save Discord Setup',
    'discordSetup.successSaved': 'Discord setup saved. The token is stored locally and never exposed in API responses.',
    'discordSetup.errorSave': 'Failed to save Discord setup',
    'discordSetup.errorApiUnavailable': 'Discord setup API client is unavailable',
    'discordSetup.errorTokenRequired': 'Discord bot token is required',
    'settings.channel.discord.phase.idle': 'Idle',
    'settings.channel.discord.phase.connecting': 'Connecting',
    'settings.channel.discord.phase.ready': 'Ready',
    'settings.channel.discord.phase.running': 'Running',
    'settings.channel.discord.phase.error': 'Error',
    'settings.channel.discord.phase.closed': 'Closed',
    'settings.channel.discord.playbackState.idle': 'Idle',
    'settings.channel.discord.playbackState.playing': 'Playing',
    'settings.channel.discord.playbackState.stopped': 'Stopped',
    'settings.channel.discord.playbackState.error': 'Error',
    'settings.channel.discord.speakingState.idle': 'Idle',
    'settings.channel.discord.speakingState.speaking': 'Speaking',
    'settings.channel.discord.speakingState.silent': 'Silent',
    'settings.channel.discord.speakingState.error': 'Error',
    'channelControl.start': 'Start',
    'channelControl.stop': 'Stop',
    'channelControl.started': '{name} started.',
    'channelControl.stopped': '{name} stopped.',
    'channelControl.errorStart': 'Failed to start {name}',
    'channelControl.errorStop': 'Failed to stop {name}',
    'channelControl.startApiUnavailable': 'Channels start API client is unavailable',
    'channelControl.stopApiUnavailable': 'Channels stop API client is unavailable',
    'settings.form.apiKey': 'API Key',
    'settings.form.apiUrl': 'API URL / Ollama Host',
    'settings.form.backend': 'Backend',
    'settings.form.device': 'Device',
    'settings.form.language': 'Language',
    'settings.form.model': 'Model',
    'settings.form.modelName': 'Model name',
    'settings.form.speed': 'Speed',
    'settings.form.voice': 'Voice',
    'settings.learning.autoExtract': 'Auto-extract Skill',
    'settings.learning.autoSyncFilesystem': 'Auto-sync SKILL.md',
    'settings.learning.description': 'Trajectories, skill library, sessions, and retention policies',
    'settings.learning.enable': 'Enable learning',
    'settings.learning.errorSave': 'Failed to save learning settings',
    'settings.learning.improvementThreshold': 'Improvement threshold',
    'settings.learning.maxSkills': 'Max skills',
    'settings.learning.minSteps': 'Minimum steps',
    'settings.learning.minToolCalls': 'Minimum tool calls',
    'settings.learning.placeholder.pluginsDir': 'e.g. ~/.mochi/plugins',
    'settings.learning.placeholder.sessionsDir': 'e.g. ~/.mochi/sessions',
    'settings.learning.placeholder.skillsDir': 'e.g. ~/.mochi/skills',
    'settings.learning.placeholder.workspaceDir': 'e.g. ~/.mochi',
    'settings.learning.pluginsDir': 'Plugins backend directory',
    'settings.learning.retentionDays': 'Trajectory retention days',
    'settings.learning.sessionsDir': 'Sessions backend directory',
    'settings.learning.skillsDir': 'Skills backend directory',
    'settings.learning.successSaved': 'Learning and path settings saved',
    'settings.learning.title': 'Learning Storage',
    'settings.learning.workspaceDir': 'Workspace / Trajectories backend directory',
    'settings.memory.dbPath': 'Memory DB backend path',
    'settings.memory.dbPathHelp': 'The SQLite file must be in a path accessible to the backend. A Windows/macOS browser cannot directly provide an absolute path the backend can open.',
    'settings.memory.description': 'SQLite memory store and retrieval policy',
    'settings.memory.errorSave': 'Failed to save memory settings',
    'settings.memory.placeholder.dbPath': 'e.g. ~/.mochi/memory.db or /var/lib/mochi/memory.db',
    'settings.memory.shortTermMessages': 'Short-term messages',
    'settings.memory.successSaved': 'Memory settings saved',
    'settings.memory.title': 'Memory Storage',
    'settings.modelConnection.apiKeyPlaceholderNoKey': 'Ollama usually does not need an API key',
    'settings.modelConnection.description': 'Successful connections are added to the available model list. Connect Ollama, OpenAI-compatible APIs, or scan backend-readable local GGUF / safetensors weights. API keys are not returned by the settings API.',
    'settings.modelConnection.errorConfigure': 'Failed to configure model',
    'settings.modelConnection.errorTest': 'Failed to test model connection',
    'settings.modelConnection.ggufRuntimeMissing': 'The GGUF inference runtime is not installed. Prepare a `llama.cpp Runtime` from Settings, or register an existing runtime path, before loading a GGUF model.',
    'settings.modelConnection.errorDiscover': 'Failed to read Ollama models',
    'settings.modelConnection.errorDiscoverLocal': 'Failed to scan local models',
    'settings.modelConnection.localModelPath': 'Model weight path',
    'settings.modelConnection.localModelPlaceholder': 'e.g. /srv/mochi/models/qwen2.5.gguf or /srv/mochi/models/Qwen2.5-7B-Instruct',
    'settings.modelConnection.localRootPath': 'Local model root',
    'settings.modelConnection.localRootPlaceholder': 'e.g. /srv/mochi/models or ~/.cache/huggingface/hub',
    'settings.modelConnection.refreshOllama': 'Refresh Ollama model list',
    'settings.modelConnection.scanLocal': 'Scan local models',
    'settings.modelConnection.successDiscovered': 'Loaded {count} Ollama models',
    'settings.modelConnection.successLocalDiscovered': 'Found {count} local models',
    'settings.modelConnection.successNoLocalModels': 'No supported local model was found under this path',
    'settings.modelConnection.successNoModels': 'Ollama is reachable, but it did not report any models',
    'settings.modelConnection.successTest': 'Connection test succeeded: {model}',
    'settings.modelConnection.successSwitched': 'Switched to {model}',
    'settings.modelConnection.successSwitchedPersisted': 'Switched and saved {model}',
    'settings.modelConnection.title': 'Model Connection',
    'settings.quantization.title': 'Quantization Capability',
    'settings.quantization.description': 'Shows GGUF conversion capability for a local Hugging Face model. GGUF is currently Mochi\'s cross-platform default path.',
    'settings.quantization.ggufHintTitle': 'Detected local Hugging Face / safetensors weights',
    'settings.quantization.ggufHintBody': 'The selected model is not GGUF. If you want better compatibility with Mochi\'s local llama.cpp runtime, converting it to GGUF is usually the safer path.',
    'settings.quantization.ggufHintDetail': 'This also applies to FP8 safetensors weights. They may already be quantized, but some GPU / runtime combinations still fall back to higher-precision loading.',
    'settings.quantization.loading': 'Loading quantization capability…',
    'settings.quantization.errorLoad': 'Failed to load quantization capability',
    'settings.quantization.supportedLabel': 'Availability',
    'settings.quantization.supportedValue': 'Plannable',
    'settings.quantization.conditionalValue': 'Conditionally available',
    'settings.quantization.unsupportedValue': 'Not yet available',
    'settings.quantization.reasonLabel': 'Reason',
    'settings.quantization.warningsLabel': 'Warnings',
    'settings.quantization.optionsLabel': 'Available quantization options',
    'settings.quantization.suggestedDefaultLabel': 'Suggested default',
    'settings.quantization.convertAction': 'Convert to GGUF',
    'settings.quantization.convertSuccess': 'GGUF conversion completed',
    'settings.quantization.convertSuccessWithPath': 'GGUF conversion completed: {path}',
    'settings.quantization.convertError': 'GGUF conversion failed',
    'settings.quantization.outputPathLabel': 'Output path',
    'settings.quantization.reasonMissing': 'The backend did not return capability data for this format.',
    'settings.quantization.hardwareTitle': 'Hardware summary',
    'settings.quantization.hardwareProvider': 'Probe provider',
    'settings.quantization.hardwareCuda': 'CUDA',
    'settings.quantization.hardwareGpuCount': 'GPU count',
    'settings.quantization.hardwareVram': 'VRAM',
    'settings.quantization.hardwarePrimaryGpu': 'Primary GPU',
    'settings.quantization.hardwareWarningsLabel': 'Hardware probe warnings',
    'settings.quantization.status.supported': 'Supported',
    'settings.quantization.status.unsupported': 'Unsupported',
    'settings.quantization.status.conditional': 'Conditional',
    'settings.quantization.status.unknown': 'Unknown',
    'settings.runtime.localRuntimeTitle': 'llama.cpp Runtime',
    'settings.runtime.localRuntimeDescription': 'GGUF model loading and GGUF conversion both depend on a working llama.cpp runtime.',
    'settings.runtime.localRuntimeLoading': 'Loading llama.cpp runtime status...',
    'settings.runtime.localRuntimeErrorLoad': 'Failed to load llama.cpp runtime status',
    'settings.runtime.localRuntimeInstallAction': 'Install recommended',
    'settings.runtime.localRuntimeRegisterAction': 'Use existing path',
    'settings.runtime.localRuntimeInstallPrepared': 'Managed runtime directory prepared.',
    'settings.runtime.localRuntimeInstallError': 'Failed to prepare llama.cpp runtime',
    'settings.runtime.localRuntimeRegistered': 'Registered existing llama.cpp path.',
    'settings.runtime.localRuntimeRegisterError': 'Failed to register existing llama.cpp path',
    'settings.runtime.localRuntimeExistingPath': 'Existing llama.cpp path',
    'settings.runtime.localRuntimeExistingPathPlaceholder': 'e.g. /opt/llama.cpp or /workspace/llama.cpp',
    'settings.runtime.localRuntimeExistingPathRequired': 'Enter an existing llama.cpp path first',
    'settings.runtime.localRuntimeSource': 'Source',
    'settings.runtime.localRuntimeVersion': 'Version',
    'settings.runtime.localRuntimeRoot': 'Root',
    'settings.runtime.localRuntimeMissing': 'Missing components',
    'settings.runtime.localRuntimeBlocked': 'Prepare the llama.cpp runtime first',
    'settings.runtime.localRuntimeState.ready': 'Ready',
    'settings.runtime.localRuntimeState.degraded': 'Degraded',
    'settings.runtime.localRuntimeState.missing': 'Missing',
    'settings.runtime.localRuntimeState.not_installed': 'Not installed',
    'settings.runtime.localRuntimeState.manual_setup_required': 'Manual setup',
    'settings.runtime.localRuntimeState.incompatible': 'Incompatible',
    'settings.runtime.localRuntimeState.installing': 'Installing',
    'settings.runtime.localRuntimeState.unknown': 'Unknown',
    'settings.preferences.timezone.autoWithZone': 'Browser default ({timezone})',
    'settings.provider.anthropic.description': 'Claude OpenAI SDK compatibility endpoint',
    'settings.provider.anthropic.note': 'Uses the Anthropic OpenAI-compatible layer. Native Anthropic API support is not implemented yet.',
    'settings.provider.gemini.description': 'Google Gemini OpenAI-compatible endpoint',
    'settings.provider.gemini.note': 'Uses the official Gemini OpenAI-compatible endpoint.',
    'settings.provider.local.description': 'Load backend-local GGUF or Hugging Face safetensors weights directly',
    'settings.provider.local.note': 'The path must be readable by the Mochi backend process. MVP loads models in the API process, so large models may require substantial RAM/VRAM.',
    'settings.provider.ollama.description': 'Local Ollama HTTP API',
    'settings.provider.ollama.note': 'After connection, /api/tags is read and the model dropdown is updated.',
    'settings.provider.openaiCompat.description': 'OpenAI-compatible Chat Completions or Responses endpoint',
    'settings.provider.openaiCompat.note': 'Stopping at /v1 uses /chat/completions; a full /responses or /chat/completions endpoint is used as-is.',
    'settings.provider.sglang.description': 'External SGLang OpenAI-compatible endpoint',
    'settings.provider.sglang.note': 'Connect to an existing SGLang server that exposes an OpenAI-compatible /v1 API. Mochi does not manage the runtime or startup flags here.',
    'settings.provider.tensorrtLlm.description': 'External TensorRT-LLM OpenAI-compatible endpoint',
    'settings.provider.tensorrtLlm.note': 'Connect to an existing TensorRT-LLM serve endpoint that exposes an OpenAI-compatible /v1 API. Mochi does not manage the runtime or startup flags here.',
    'settings.provider.openaiCodex.description': 'OpenAI Codex backend-api transport backed by ChatGPT OAuth',
    'settings.provider.openaiCodex.note': 'This path does not use an API key. Import the local Codex CLI ChatGPT login first, then connect the model.',
    'settings.voice.description': 'STT/TTS runtime and local model storage',
    'settings.voice.downloadMissing': 'Prepare downloads when models are missing',
    'settings.voice.enable': 'Enable voice',
    'settings.voice.enableHelp': 'New voice sessions will use updated settings after saving.',
    'settings.voice.errorImport': 'Failed to import model',
    'settings.voice.errorSave': 'Failed to save voice settings',
    'settings.voice.importSuccess': 'Imported {count} files ({bytes}) and filled the backend path. Save Voice Pipeline to apply the setting.',
    'settings.voice.placeholder.sttCacheDir': 'e.g. ~/.cache/mochi/stt or /var/lib/mochi/models/stt',
    'settings.voice.placeholder.sttModelPath': 'e.g. /srv/mochi/models/whisper/large-v3 or a single model file',
    'settings.voice.saveSuccess': 'Voice settings saved',
    'settings.voice.saveSuccessWithStatus': 'Voice settings saved. Model status: {status}',
    'settings.voice.sttCacheDir': 'STT backend model cache/download directory',
    'settings.voice.sttCacheDirHelp': 'This is the server path the backend reads and writes. A WSL backend uses Linux/WSL paths, not browser-local paths.',
    'settings.voice.sttModelPath': 'STT backend local model path (optional override)',
    'settings.voice.sttModelPathHelp': 'Manually enter an existing backend model file or directory. Large folders are no longer enumerated through the browser to avoid browser crashes.',
    'settings.voice.sttTitle': 'Speech to Text',
    'settings.voice.ttsTitle': 'Text to Speech',
    'settings.voice.title': 'Voice Pipeline',
    'settings.voice.replyModelTitle': 'Reply Model',
    'settings.voice.replyModelMode': 'Mode',
    'settings.voice.replyModelId': 'Model ID (fixed mode)',
    'settings.voice.replyModelHelp': 'Use global mode to follow global model settings, or fixed mode to force a specific reply model for voice sessions.',
    'settings.voice.sessionModeTitle': 'Session Mode',
    'settings.voice.sessionMode': 'Mode',
    'settings.voice.sessionModeHelp': 'Shared mode attaches voice turns to the active chat session. Isolated mode uses dedicated voice session handling.',
    'settings.voice.voicePacksTitle': 'Voice Packs',
    'settings.voice.voicePacksUpload': 'Upload',
    'settings.voice.voicePacksRegisterPath': 'Register path',
    'settings.voice.voicePacksRefresh': 'Refresh',
    'settings.voice.voicePacksPathLabel': 'Register existing backend path',
    'settings.voice.voicePacksPathPlaceholder': 'e.g. /srv/mochi/voices or H:\\voices\\pack',
    'settings.voice.voicePacksUnavailable': 'Voice pack management routes are not available on this backend yet.',
    'settings.voice.runtimeLoading': 'Loading runtime status...',
    'settings.voice.runtimeLoaded': 'loaded',
    'settings.voice.runtimeEnabled': 'enabled',
    'settings.voice.voicePacksUnavailableBadge': 'Unavailable',
    'settings.voice.voicePacksCount': '{count} voices',
    'sidebar.collapse': 'Collapse sidebar',
    'sidebar.deleteConversation': 'Delete conversation',
    'sidebar.deleteConfirm': 'Delete "{title}"?',
    'sidebar.deleteDialogAction': 'Delete',
    'sidebar.deleteDialogDescription': 'This will permanently delete "{title}".',
    'sidebar.deleteDialogTitle': 'Delete conversation?',
    'sidebar.bulkDelete': 'Bulk delete',
    'sidebar.bulkCancel': 'Cancel selection',
    'sidebar.bulkSelectAll': 'Select all visible',
    'sidebar.bulkClearAll': 'Clear selection',
    'sidebar.bulkDeleteSelected': 'Delete selected',
    'sidebar.bulkSelectedCount': '{count} selected',
    'sidebar.browseFolder': 'Browse folder',
    'sidebar.bulkDeleteDialogTitle': 'Delete selected conversations?',
    'sidebar.bulkDeleteDialogDescription': 'This will permanently delete {count} conversations.',
    'sidebar.expand': 'Expand sidebar',
    'sidebar.newChat': 'New chat',
    'sidebar.newChatShortcut': 'New chat (⌘/)',
    'sidebar.noResults': 'No matching conversations',
    'sidebar.noSessions': 'No conversation history',
    'sidebar.older': 'Older',
    'sidebar.pinned': 'Pinned',
    'sidebar.projectDirectoryPickerFailed': 'Failed to open folder picker',
    'sidebar.rename': 'Rename',
    'sidebar.renameCancel': 'Cancel rename',
    'sidebar.renameSave': 'Save name',
    'sidebar.searchPlaceholder': 'Search conversations... (⌘K)',
    'app.shortcuts.openInput': 'Focus input (⌘L)',
    'app.shortcuts.toggleSidebar': 'Toggle sidebar (⌘B)',
    'sidebar.settings': 'Settings',
    'sidebar.skills': 'Skill Library',
    'sidebar.workflows': 'Workflows',
    'sidebar.goals': 'Goals',
    'sidebar.thisWeek': 'This week',
    'sidebar.today': 'Today',
    'workflows.apiUnavailable': 'Workflows API is not available on this backend yet.',
    'workflows.backToList': 'Back to Workflows',
    'workflows.create': 'Create Workflow',
    'workflows.createDescription': 'Configure protocol and subagents. Models come from configured providers.',
    'workflows.createError': 'Unable to create Workflow.',
    'workflows.description': 'Create and monitor structured multi-agent workflows for collaboration, evidence gathering, evaluation, and scheduling.',
    'workflows.loadError': 'Unable to load Workflows.',
    'workflows.title': 'Workflows',
    'agentRuns.status.created': 'Created',
    'agentRuns.status.running': 'Running',
    'agentRuns.status.queued': 'Queued',
    'agentRuns.status.pending': 'Pending',
    'agentRuns.status.paused': 'Paused',
    'agentRuns.status.awaiting_resources': 'Awaiting resources',
    'agentRuns.status.stalled': 'Stalled',
    'agentRuns.status.partial': 'Partial',
    'agentRuns.status.degraded': 'Degraded',
    'agentRuns.status.succeeded': 'Succeeded',
    'agentRuns.status.completed': 'Completed',
    'agentRuns.status.done': 'Done',
    'agentRuns.status.failed': 'Failed',
    'agentRuns.status.cancelled': 'Cancelled',
    'agentRuns.status.error': 'Error',
    'agentRuns.badge.degraded': 'Degraded',
    'agentRuns.badge.finalizePartialReady': 'Finalize partial ready',
    'agentRuns.runPolicy.title': 'Run Policy',
    'agentRuns.runPolicy.description': 'Bound total runtime, detect stalled subagents, and define how the run should recover when resources or roles drop.',
    'agentRuns.runPolicy.preset': 'Preset',
    'agentRuns.runPolicy.presetPlaceholder': 'Select run policy preset',
    'agentRuns.runPolicy.short': 'Short task',
    'agentRuns.runPolicy.balanced': 'Balanced',
    'agentRuns.runPolicy.long': 'Long research',
    'agentRuns.runPolicy.custom': 'Custom',
    'agentRuns.runPolicy.maxWallClock': 'Max wall clock (sec)',
    'agentRuns.runPolicy.heartbeatTimeout': 'Heartbeat timeout (sec)',
    'agentRuns.runPolicy.checkpointInterval': 'Checkpoint interval',
    'agentRuns.runPolicy.maxFailuresPerRole': 'Max failures / role',
    'agentRuns.runPolicy.budgetExhausted': 'Budget exhausted',
    'agentRuns.runPolicy.budgetPlaceholder': 'Select budget action',
    'agentRuns.runPolicy.disconnect': 'Subagent disconnect',
    'agentRuns.runPolicy.disconnectPlaceholder': 'Select disconnect action',
    'agentRuns.runPolicy.budgetPause': 'Move to awaiting_resources',
    'agentRuns.runPolicy.budgetFinalizePartial': 'Finalize as partial',
    'agentRuns.runPolicy.disconnectRetryThenDegrade': 'Retry then degrade',
    'agentRuns.runPolicy.disconnectPause': 'Move to stalled',
    'agentRuns.runPolicy.disconnectFail': 'Fail run',
    'agentRuns.recentRuns.title': 'Recent Runs',
    'agentRuns.recentRuns.description': 'Open a run to inspect events, artifacts, recovery prompts, and guidance controls.',
    'agentRuns.recentRuns.empty': 'No agent runs yet.',
    'agentRuns.run.untitled': 'Untitled run',
    'agentRuns.run.noTopic': 'No topic provided.',
    'agentRuns.run.updated': 'Updated: {value}',
    'agentRuns.run.state.active': 'Active',
    'agentRuns.run.state.finished': 'Finished',
    'agentRuns.recovery.title': 'Recovery',
    'agentRuns.recovery.description': 'Current recovery state, effective run policy, and subagent health summary.',
    'agentRuns.recovery.operatorTitle': 'Operator prompt',
    'agentRuns.recovery.resumeTitle': 'Resume hint',
    'agentRuns.recovery.policyTitle': 'Run Policy',
    'agentRuns.recovery.stateTitle': 'Recovery State',
    'agentRuns.recovery.healthTitle': 'Subagent Health',
    'agentRuns.recovery.emptyHealth': 'No extra subagent health snapshot has been reported yet.',
    'agentRuns.recovery.awaiting_resources.message': 'This run is waiting on more resources. Restore models, tool approvals, data, or budget before resuming.',
    'agentRuns.recovery.awaiting_resources.resume': 'Confirm the missing resource is available, then resume the current attempt.',
    'agentRuns.recovery.stalled.message': 'This run is stalled. Check subagent health, the latest error, and detached exec status before deciding to resume.',
    'agentRuns.recovery.stalled.resume': 'If the external process or dependency has recovered, you can resume directly. Otherwise send guidance or stop the background job first.',
    'agentRuns.recovery.partial.message': 'This run retained a partial result. Review the current artifacts, then decide whether to resume with more guidance.',
    'agentRuns.recovery.partial.resume': 'Resume continues from the current context and preserved outputs, which is useful after you confirm direction.',
    'agentRuns.recovery.degraded.message': 'This run is degraded: some roles or capabilities dropped out, but the workflow can continue with reduced coverage.',
    'agentRuns.recovery.degraded.resume': 'Before resuming, decide whether degraded execution is acceptable. If not, repair dependencies or swap models first.',
    'agentRuns.recovery.default.message': 'No extra recovery instructions were reported. Inspect events and artifacts to decide the next step.',
    'agentRuns.recovery.default.resume': 'If the run is paused and prerequisites are satisfied, you can resume directly.',
    'agentRuns.recovery.operatorFallback': 'The backend did not provide an operator message, so this UI is showing a status-based recommendation.',
    'agentRuns.recovery.latestError': 'Latest error',
    'agentRuns.recovery.suggestedAction': 'Suggested action',
    'agentRuns.recovery.finalizePartialAvailable': 'The current state can be preserved as partial immediately instead of waiting on blocked resources or a stalled stage.',
    'agentRuns.recovery.finalizePartialReason': 'Finalize-partial note',
    'agentRuns.actions.finalizePartial': 'Finalize Partial',
    'agentRuns.exec.title': 'Detached Exec Jobs',
    'agentRuns.exec.description': 'Inspect background exec jobs, reattach to their latest output, or ask the runtime to stop a session.',
    'agentRuns.exec.empty': 'No detached exec jobs are currently reported.',
    'agentRuns.exec.emptyStalled': 'No detached exec jobs were reported. If the run is still stalled, inspect subagent health, the latest error, and external dependencies first.',
    'agentRuns.exec.command': 'Command',
    'agentRuns.exec.status': 'Status',
    'agentRuns.exec.liveStatus': 'Live status',
    'agentRuns.exec.leaseOwner': 'Lease owner',
    'agentRuns.exec.workdir': 'Workdir',
    'agentRuns.exec.logPath': 'Log path',
    'agentRuns.exec.approval': 'Approval',
    'agentRuns.exec.background': 'Background',
    'agentRuns.exec.timeout': 'Timeout',
    'agentRuns.exec.pid': 'PID',
    'agentRuns.exec.reattachSupported': 'Reattach supported',
    'agentRuns.exec.stopStatus': 'Stop result',
    'agentRuns.exec.reattachStatus': 'Reattach result',
    'agentRuns.exec.noCommand': 'No command text was reported.',
    'agentRuns.exec.noSnapshot': 'No session output snapshot has been fetched yet.',
    'agentRuns.exec.stdout': 'Stdout',
    'agentRuns.exec.stderr': 'Stderr',
    'agentRuns.exec.stdoutEmpty': 'No stdout captured yet.',
    'agentRuns.exec.stderrEmpty': 'No stderr captured yet.',
    'agentRuns.exec.snapshotTitle': 'Latest Session Snapshot',
    'agentRuns.exec.snapshotDescription': 'This snapshot reflects the most recent poll / reattach / stop action.',
    'agentRuns.exec.rawPayload': 'Raw payload',
    'agentRuns.exec.action.poll': 'Poll',
    'agentRuns.exec.action.reattach': 'Reattach',
    'agentRuns.exec.action.stop': 'Stop',
    'agentRuns.exec.action.unavailable': 'Unavailable',
    'agentRuns.exec.boolean.yes': 'Yes',
    'agentRuns.exec.boolean.no': 'No',
    'agentRuns.exec.status.running': 'Running',
    'agentRuns.exec.status.pending': 'Pending',
    'agentRuns.exec.status.queued': 'Queued',
    'agentRuns.exec.status.completed': 'Completed',
    'agentRuns.exec.status.succeeded': 'Succeeded',
    'agentRuns.exec.status.failed': 'Failed',
    'agentRuns.exec.status.error': 'Error',
    'agentRuns.exec.status.stopped': 'Stopped',
    'agentRuns.exec.status.unavailable': 'Unavailable',
    'skills.card.created': 'Created',
    'skills.card.delete': 'Delete {name}',
    'skills.card.noDescription': 'No description',
    'skills.card.success': 'Success',
    'skills.card.unknown': 'Unknown',
    'skills.card.untagged': 'untagged',
    'skills.card.used': 'Used',
    'skills.empty': 'No displayable skills yet',
    'skills.errorDelete': 'Failed to delete skill',
    'skills.errorLoad': 'Failed to load skills',
    'skills.loaded': 'Loaded',
    'skills.noSearchResults': 'No skills match "{query}"',
    'skills.refresh': 'Refresh',
    'skills.searchPlaceholder': 'Search name, description, or tags',
    'skills.subtitle': 'Backend skill index and usage summary',
    'skills.tabs.all': 'All',
    'skills.tabs.highestSuccess': 'High success',
    'skills.tabs.mostUsed': 'Most used',
    'skills.tabs.recent': 'Recent',
    'skills.title': 'Skill Library',
    'settings.title': 'Settings',
    'settings.subtitle': 'Backend configuration summary and service readiness',
    'settings.errorLoadFailed': 'Failed to load settings',
    'settings.badge.connected': 'Connected',
    'settings.badge.partial': 'Partial',
    'settings.stats.primaryModel': 'Primary Model',
    'settings.stats.modelEndpoints': 'Model Endpoints',
    'settings.stats.channels': 'Channels',
    'settings.stats.webSurface': 'Web Surface',
    'settings.stats.configured': 'Configured',
    'settings.stats.notReported': 'Not reported',
    'settings.tabs.model': 'Model',
    'settings.tabs.inference': 'Inference',
    'settings.tabs.voice': 'Voice',
    'settings.tabs.memory': 'Memory',
    'settings.tabs.learning': 'Learning',
    'settings.tabs.security': 'Security',
    'settings.tabs.channels': 'Channels',
    'settings.tabs.web': 'Web',
    'settings.action.saveSecurity': 'Save Security',
    'settings.security.title': 'Autonomy & Safety',
    'settings.security.description': 'Tune autonomy level and the default approval policy for exec and file tools.',
    'settings.security.autonomyMode': 'Autonomy mode',
    'settings.security.autonomyMode.strict': 'Strict',
    'settings.security.autonomyMode.trusted_workspace': 'Trusted workspace',
    'settings.security.autonomyMode.auto_review': 'Auto review',
    'settings.security.autonomyMode.high_autonomy': 'High autonomy',
    'settings.security.requireFileWriteApproval': 'Require approval for file writes',
    'settings.security.requireExecApproval': 'Require approval for exec runtime',
    'settings.security.execDefaultTimeoutSec': 'Default exec timeout (sec)',
    'settings.security.execSessionOutputLimit': 'Exec output tail limit (chars)',
    'settings.security.agentRunDefaultMaxWallClockSec': 'Default agent-run max wall clock (sec)',
    'settings.security.agentRunDefaultMaxWallClockSecPlaceholder': 'Leave blank to disable the default deadline',
    'settings.security.agentRunDefaultOnBudgetExhausted': 'Default agent-run budget action',
    'settings.security.agentRunDefaultOnBudgetExhausted.pause': 'Move to awaiting_resources',
    'settings.security.agentRunDefaultOnBudgetExhausted.finalize_partial': 'Finish as partial',
    'settings.security.agentRunDefaultHeartbeatTimeoutSec': 'Default agent-run heartbeat timeout (sec)',
    'settings.security.agentRunDefaultHeartbeatTimeoutSecPlaceholder': 'Leave blank to disable stalled detection',
    'settings.security.agentRunDefaultCheckpointIntervalSteps': 'Default checkpoint interval (steps)',
    'settings.security.agentRunDefaultMaxSubagentFailuresPerRole': 'Default max subagent failures per role',
    'settings.security.agentRunDefaultOnSubagentDisconnect': 'Default subagent disconnect action',
    'settings.security.agentRunDefaultOnSubagentDisconnect.retry_then_degrade': 'Retry then degrade',
    'settings.security.agentRunDefaultOnSubagentDisconnect.pause': 'Move to stalled',
    'settings.security.agentRunDefaultOnSubagentDisconnect.fail': 'Fail the run',
    'settings.security.fileScope': 'File operation scope',
    'settings.security.fileScope.workspace': 'Workspace only',
    'settings.security.fileScope.any': 'Any path (higher risk)',
    'settings.security.maxWriteSizeMb': 'Max write size per operation (MB)',
    'settings.security.undoMaxSizeMb': 'Max undo content size (MB)',
    'settings.security.successSaved': 'Security settings saved',
    'settings.security.errorSave': 'Failed to save security settings',
    'settings.section.modelConfig': 'Model Config',
    'settings.section.discoveredModels': 'Discovered Models',
    'settings.savedModels.title': 'Saved Models',
    'settings.savedModels.edit': 'Edit',
    'settings.savedModels.delete': 'Delete',
    'settings.savedModels.save': 'Save',
    'settings.savedModels.cancel': 'Cancel',
    'settings.savedModels.apiKeyHint': 'Leave empty to keep existing API key; enter a new value to replace it.',
    'settings.savedModels.none': 'No saved models yet',
    'settings.savedModels.errorUpdate': 'Failed to update model entry',
    'settings.savedModels.errorDelete': 'Failed to delete model entry',
    'settings.savedModels.errorTest': 'Failed to test saved model connection',
    'settings.savedModels.successTest': 'Connection test succeeded: {model}',
    'settings.savedModels.successUpdate': 'Model entry updated',
    'settings.savedModels.successDelete': 'Model entry deleted',
    'settings.section.voicePipeline': 'Voice Pipeline',
    'settings.section.runtime': 'Runtime',
    'settings.section.memory': 'Memory',
    'settings.section.learning': 'Learning',
    'settings.section.channels': 'Channels',
    'settings.section.web': 'Web',
    'settings.summary.modelConfig': 'Only non-sensitive backend fields are shown',
    'settings.summary.discoveredModels': 'Available model summary from the models API',
    'settings.summary.voicePipeline': 'Summary of STT, TTS, and I/O devices',
    'settings.summary.memory': 'Storage backend, paths, and retention summary',
    'settings.summary.learning': 'Skill extraction and trajectory retention summary',
    'settings.summary.channels': 'Channel status only, without exposing bot tokens or API keys',
    'settings.summary.web': 'Web service and frontend integration summary',
    'settings.runtime.voiceFields': 'Voice fields',
    'settings.runtime.settingsSource': 'Settings source',
    'settings.runtime.backend': 'Backend',
    'settings.runtime.unavailable': 'Unavailable',
    'settings.preferences.title': 'UI Preferences',
    'settings.preferences.description': 'Adjust language, appearance, font size, and time zone. Preferences are stored in this browser only.',
    'settings.preferences.language': 'UI language',
    'settings.preferences.appearance': 'Appearance mode',
    'settings.preferences.codeTheme': 'Code theme',
    'settings.preferences.fontSize': 'Font size',
    'settings.preferences.timezone': 'Time zone',
    'settings.preferences.appearanceHelp': 'Appearance can follow your system, or be pinned to dark/light.',
    'settings.preferences.codeThemeHelp': 'The selected code theme applies to every code block across the WebGUI.',
    'settings.preferences.fontHelp': 'Font size affects readability across the main WebGUI.',
    'settings.preferences.timezoneHelp': 'Time zone affects display only. Storage and API timestamps remain UTC.',
    'settings.preferences.codeThemePreview': 'Code theme preview',
    'settings.preferences.language.default': 'Default / Auto-detect',
    'settings.preferences.language.zhTW': 'Traditional Chinese',
    'settings.preferences.language.en': 'English',
    'settings.preferences.appearance.system': 'System',
    'settings.preferences.appearance.dark': 'Dark',
    'settings.preferences.appearance.light': 'Light',
    'settings.preferences.font.compact': 'Compact',
    'settings.preferences.font.default': 'Default',
    'settings.preferences.font.large': 'Large',
    'settings.preferences.timezone.auto': 'Browser default',
    'settings.preferences.timezone.utc': 'UTC',
    'settings.emptyState': 'No settings summary is available from backend yet.',
    'settings.noSummary': 'No summary available',
    'settings.modelFallback': 'Model',
  },
}

function normalizeExplicitLanguage(value: unknown): UILanguage | null {
  if (typeof value !== 'string') {
    return null
  }

  const normalized = value.trim().toLowerCase()
  if (
    normalized === 'zh'
    || normalized === 'zh-tw'
    || normalized === 'zh_tw'
    || normalized.startsWith('zh-hant')
  ) {
    return 'zh-TW'
  }
  if (normalized === 'en' || normalized === 'en-us' || normalized === 'en_us') {
    return 'en'
  }

  return null
}

function resolveBrowserLanguage(): UILanguage {
  if (typeof window === 'undefined') {
    return DEFAULT_LANGUAGE
  }

  const candidates = Array.isArray(window.navigator.languages) && window.navigator.languages.length > 0
    ? window.navigator.languages
    : [window.navigator.language]

  for (const candidate of candidates) {
    const language = normalizeExplicitLanguage(candidate)
    if (language) {
      return language
    }
  }

  return DEFAULT_LANGUAGE
}

export function resolveLocale(language: UILanguage): string {
  return LANGUAGE_LOCALE[language] ?? LANGUAGE_LOCALE.en
}

function resolveLanguageFromMode(languageMode: UILanguageMode, browserLanguage: UILanguage): UILanguage {
  return languageMode === AUTO_LANGUAGE ? browserLanguage : languageMode
}

function resolveAppearanceFromMedia(): UIAppearance {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return DEFAULT_APPEARANCE
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function resolveAppearance(mode: UIAppearanceMode, systemAppearance: UIAppearance): UIAppearance {
  return mode === SYSTEM_APPEARANCE ? systemAppearance : mode
}

function resolveBrowserTimeZone(): string | undefined {
  if (typeof Intl === 'undefined') {
    return undefined
  }
  const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone
  if (typeof timeZone !== 'string' || timeZone.trim().length === 0) {
    return undefined
  }
  return timeZone
}

export function resolveDisplayTimeZone(timezone: UITimezone): string | undefined {
  if (!timezone || timezone === AUTO_TIMEZONE) {
    return resolveBrowserTimeZone()
  }
  return timezone
}

type UIPreferences = {
  languageMode: UILanguageMode
  fontSize: UIFontSize
  appearanceMode: UIAppearanceMode
  codeTheme: UICodeTheme
  timezone: UITimezone
}

function buildDefaultPreferences(): UIPreferences {
  return {
    languageMode: DEFAULT_LANGUAGE_MODE,
    fontSize: DEFAULT_FONT_SIZE,
    appearanceMode: DEFAULT_APPEARANCE_MODE,
    codeTheme: DEFAULT_CODE_THEME,
    timezone: AUTO_TIMEZONE,
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function normalizeLanguageMode(value: unknown): UILanguageMode {
  if (value === AUTO_LANGUAGE) {
    return AUTO_LANGUAGE
  }

  return normalizeExplicitLanguage(value) ?? DEFAULT_LANGUAGE_MODE
}

function normalizeAppearanceMode(value: unknown): UIAppearanceMode {
  if (value === 'system' || value === 'dark' || value === 'light') {
    return value
  }
  return DEFAULT_APPEARANCE_MODE
}

function normalizeFontSize(value: unknown): UIFontSize {
  if (value === 'compact' || value === 'default' || value === 'large') {
    return value
  }
  return DEFAULT_FONT_SIZE
}

function normalizeTimezone(value: unknown): UITimezone {
  if (typeof value !== 'string') {
    return AUTO_TIMEZONE
  }
  const next = value.trim()
  return next.length > 0 ? next : AUTO_TIMEZONE
}

function parseStoredPreferences(raw: string | null): UIPreferences {
  const defaults = buildDefaultPreferences()
  if (!raw) {
    return defaults
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return defaults
  }

  // Backward compatibility: early payload could be a plain language string.
  if (typeof parsed === 'string') {
    const explicitLanguage = normalizeExplicitLanguage(parsed)
    if (!explicitLanguage) {
      return defaults
    }

    return {
      ...defaults,
      languageMode: explicitLanguage,
    }
  }

  if (!isRecord(parsed)) {
    return defaults
  }

  return {
    languageMode: normalizeLanguageMode(
      parsed.languageMode
      ?? parsed.language_mode
      ?? parsed.language
      ?? parsed.lang
      ?? parsed.locale
    ),
    fontSize: normalizeFontSize(parsed.fontSize ?? parsed.font_size ?? parsed.font),
    appearanceMode: normalizeAppearanceMode(
      parsed.appearanceMode
      ?? parsed.appearance_mode
      ?? parsed.appearance
      ?? parsed.theme
      ?? parsed.colorScheme
    ),
    codeTheme: normalizeCodeTheme(parsed.codeTheme ?? parsed.code_theme ?? parsed.syntaxTheme),
    timezone: normalizeTimezone(parsed.timezone ?? parsed.time_zone ?? parsed.tz),
  }
}

function readStoredPreferences(): UIPreferences {
  if (typeof window === 'undefined') {
    return buildDefaultPreferences()
  }

  try {
    return parseStoredPreferences(window.localStorage.getItem(STORAGE_KEY))
  } catch {
    return buildDefaultPreferences()
  }
}

function persistPreferences(preferences: UIPreferences) {
  if (typeof window === 'undefined') {
    return
  }

  try {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        languageMode: preferences.languageMode,
        fontSize: preferences.fontSize,
        appearanceMode: preferences.appearanceMode,
        codeTheme: preferences.codeTheme,
        timezone: preferences.timezone,
      })
    )
  } catch {
    // Ignore storage write failures (e.g. private mode / quota).
  }
}

function arePreferencesEqual(left: UIPreferences, right: UIPreferences): boolean {
  return (
    left.languageMode === right.languageMode
    && left.fontSize === right.fontSize
    && left.appearanceMode === right.appearanceMode
    && left.codeTheme === right.codeTheme
    && left.timezone === right.timezone
  )
}

function updateThemeColor(appearance: UIAppearance) {
  if (typeof document === 'undefined') {
    return
  }

  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute('content', THEME_COLOR_BY_APPEARANCE[appearance])
}

function applyDocumentPreferences(
  language: UILanguage,
  fontSize: UIFontSize,
  appearance: UIAppearance,
  codeTheme: UICodeTheme
) {
  if (typeof document === 'undefined') {
    return
  }
  document.documentElement.lang = language
  document.documentElement.dataset.fontSize = fontSize
  document.documentElement.dataset.theme = appearance
  document.documentElement.dataset.codeTheme = codeTheme
  document.documentElement.style.colorScheme = appearance
  updateThemeColor(appearance)
}

const I18nContext = React.createContext<I18nContextValue | null>(null)
const useIsomorphicLayoutEffect =
  typeof window === 'undefined' ? React.useEffect : React.useLayoutEffect

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [preferences, setPreferences] = React.useState<UIPreferences>(
    () => buildDefaultPreferences()
  )
  const [browserLanguage, setBrowserLanguage] = React.useState<UILanguage>(DEFAULT_LANGUAGE)
  const [systemAppearance, setSystemAppearance] = React.useState<UIAppearance>(DEFAULT_APPEARANCE)
  const [initialized, setInitialized] = React.useState(false)
  const { languageMode, fontSize, appearanceMode, codeTheme, timezone } = preferences
  const language = React.useMemo(
    () => resolveLanguageFromMode(languageMode, browserLanguage),
    [browserLanguage, languageMode]
  )
  const appearance = React.useMemo(
    () => resolveAppearance(appearanceMode, systemAppearance),
    [appearanceMode, systemAppearance]
  )

  useIsomorphicLayoutEffect(() => {
    const nextBrowserLanguage = resolveBrowserLanguage()
    const nextSystemAppearance = resolveAppearanceFromMedia()
    const nextPreferences = readStoredPreferences()
    const nextLanguage = resolveLanguageFromMode(nextPreferences.languageMode, nextBrowserLanguage)
    const nextAppearance = resolveAppearance(nextPreferences.appearanceMode, nextSystemAppearance)

    setBrowserLanguage(nextBrowserLanguage)
    setSystemAppearance(nextSystemAppearance)
    setPreferences(nextPreferences)
    applyDocumentPreferences(nextLanguage, nextPreferences.fontSize, nextAppearance, nextPreferences.codeTheme)
    setInitialized(true)
  }, [])

  React.useEffect(() => {
    if (!initialized) {
      return
    }
    applyDocumentPreferences(language, fontSize, appearance, codeTheme)
    persistPreferences(preferences)
  }, [appearance, codeTheme, fontSize, initialized, language, preferences, timezone])

  React.useEffect(() => {
    if (!initialized || appearanceMode !== 'system' || typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
      return
    }

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleChange = (event: MediaQueryListEvent) => {
      setSystemAppearance(event.matches ? 'dark' : 'light')
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => {
      mediaQuery.removeEventListener('change', handleChange)
    }
  }, [appearanceMode, fontSize, initialized, language])

  React.useEffect(() => {
    if (!initialized || typeof window === 'undefined') {
      return
    }

    const handleLanguageChange = () => {
      setBrowserLanguage(resolveBrowserLanguage())
    }

    window.addEventListener('languagechange', handleLanguageChange)
    return () => {
      window.removeEventListener('languagechange', handleLanguageChange)
    }
  }, [initialized])

  React.useEffect(() => {
    if (!initialized || typeof window === 'undefined') {
      return
    }

    const handleStorage = (event: StorageEvent) => {
      if (event.storageArea !== window.localStorage || event.key !== STORAGE_KEY) {
        return
      }
      const nextPreferences = parseStoredPreferences(event.newValue)
      const nextLanguage = resolveLanguageFromMode(nextPreferences.languageMode, browserLanguage)
      const nextAppearance = resolveAppearance(nextPreferences.appearanceMode, systemAppearance)
      setPreferences((previous) => (
        arePreferencesEqual(previous, nextPreferences) ? previous : nextPreferences
      ))
      applyDocumentPreferences(nextLanguage, nextPreferences.fontSize, nextAppearance, nextPreferences.codeTheme)
    }

    window.addEventListener('storage', handleStorage)
    return () => {
      window.removeEventListener('storage', handleStorage)
    }
  }, [browserLanguage, initialized, systemAppearance])

  const setLanguageMode = React.useCallback(
    (nextMode: UILanguageMode) => {
      const normalizedMode = normalizeLanguageMode(nextMode)
      setPreferences((previous) => (
        previous.languageMode === normalizedMode
          ? previous
          : { ...previous, languageMode: normalizedMode }
      ))
    },
    []
  )

  const setLanguage = React.useCallback(
    (nextLanguage: UILanguage) => {
      setLanguageMode(nextLanguage)
    },
    [setLanguageMode]
  )

  const setFontSize = React.useCallback(
    (nextFontSize: UIFontSize) => {
      setPreferences((previous) => (
        previous.fontSize === nextFontSize
          ? previous
          : { ...previous, fontSize: nextFontSize }
      ))
    },
    []
  )

  const setAppearanceMode = React.useCallback(
    (nextMode: UIAppearanceMode) => {
      const normalizedMode = normalizeAppearanceMode(nextMode)
      setPreferences((previous) => (
        previous.appearanceMode === normalizedMode
          ? previous
          : { ...previous, appearanceMode: normalizedMode }
      ))
    },
    []
  )

  const setCodeTheme = React.useCallback(
    (nextTheme: UICodeTheme) => {
      const normalizedTheme = normalizeCodeTheme(nextTheme)
      setPreferences((previous) => (
        previous.codeTheme === normalizedTheme
          ? previous
          : { ...previous, codeTheme: normalizedTheme }
      ))
    },
    []
  )

  const setTimezone = React.useCallback(
    (nextTimezone: UITimezone) => {
      const normalizedTimezone = normalizeTimezone(nextTimezone)
      setPreferences((previous) => (
        previous.timezone === normalizedTimezone
          ? previous
          : { ...previous, timezone: normalizedTimezone }
      ))
    },
    []
  )

  const t = React.useCallback(
    (key: TranslationKey, values?: TranslationValues): string => {
      return interpolate(messages[language][key] ?? messages.en[key] ?? key, values)
    },
    [language]
  )

  const locale = React.useMemo(() => resolveLocale(language), [language])
  const resolvedTimeZone = React.useMemo(
    () => resolveDisplayTimeZone(timezone),
    [timezone]
  )

  const value = React.useMemo<I18nContextValue>(
    () => ({
      languageMode,
      setLanguageMode,
      language,
      setLanguage,
      locale,
      fontSize,
      setFontSize,
      appearanceMode,
      setAppearanceMode,
      appearance,
      codeTheme,
      setCodeTheme,
      timezone,
      setTimezone,
      resolvedTimeZone,
      t,
    }),
    [appearance, appearanceMode, codeTheme, fontSize, language, languageMode, locale, resolvedTimeZone, setAppearanceMode, setCodeTheme, setFontSize, setLanguage, setLanguageMode, setTimezone, t, timezone]
  )

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>
}

export function useI18n() {
  const context = React.useContext(I18nContext)
  if (!context) {
    throw new Error('useI18n must be used within I18nProvider')
  }
  return context
}
