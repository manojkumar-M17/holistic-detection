export interface Incident {
  id: number
  student_id: number | string
  activity: string
  timestamp: string
  screenshot_path?: string | null
  severity?: string | null
  category?: string | null
  risk_score?: number | null
  status?: string | null
}

export interface DashboardStats {
  active_students: number
  active_alerts: number
  total_logged_alerts: number
  unique_students_flagged: number
  severity_counts: Record<string, number>
  status_counts: Record<string, number>
  category_counts: Record<string, number>
  student_risk_scores: Record<string, number>
  audio_metrics: Record<string, number | boolean | string | null>
}

export interface DetectionConfig {
  yaw_threshold: number
  pitch_threshold: number
  suspicious_duration: number
  enable_object_detection: boolean
  enable_absence_detection: boolean
  enable_audio_detection: boolean
  audio_threshold: number
}

export class ApiError extends Error {
  readonly status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, '')
let apiAccessKey = ''
const ACCESS_KEY_CHANGE_EVENT = 'api-access-key-change'

export function getApiBaseUrl() {
  return apiBaseUrl
}

export function setApiAccessKey(token: string) {
  apiAccessKey = token.trim()
  window.dispatchEvent(new Event(ACCESS_KEY_CHANGE_EVENT))
}

export function clearApiAccessKey() {
  apiAccessKey = ''
  window.dispatchEvent(new Event(ACCESS_KEY_CHANGE_EVENT))
}

export function hasApiAccessKey() {
  return apiAccessKey.length > 0
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isNumberRecord(value: unknown): value is Record<string, number> {
  return isRecord(value) && Object.values(value).every((item) => typeof item === 'number' && Number.isFinite(item))
}

function isAudioMetrics(value: unknown): value is DashboardStats['audio_metrics'] {
  return isRecord(value) && Object.values(value).every((item) =>
    item === null || typeof item === 'string' || typeof item === 'boolean' || (typeof item === 'number' && Number.isFinite(item)),
  )
}

function isIncident(value: unknown): value is Incident {
  return isRecord(value)
    && typeof value.id === 'number'
    && Number.isInteger(value.id)
    && (typeof value.student_id === 'number' || typeof value.student_id === 'string')
    && typeof value.activity === 'string'
    && typeof value.timestamp === 'string'
    && (value.risk_score === undefined || value.risk_score === null || (typeof value.risk_score === 'number' && Number.isFinite(value.risk_score)))
    && (value.screenshot_path === undefined || value.screenshot_path === null || typeof value.screenshot_path === 'string')
    && (value.severity === undefined || value.severity === null || typeof value.severity === 'string')
    && (value.category === undefined || value.category === null || typeof value.category === 'string')
    && (value.status === undefined || value.status === null || typeof value.status === 'string')
}

function parseStats(value: unknown): DashboardStats {
  if (
    !isRecord(value)
    || typeof value.active_students !== 'number'
    || typeof value.active_alerts !== 'number'
    || typeof value.total_logged_alerts !== 'number'
    || typeof value.unique_students_flagged !== 'number'
    || !isNumberRecord(value.severity_counts)
    || !isNumberRecord(value.status_counts)
    || !isNumberRecord(value.category_counts)
    || !isNumberRecord(value.student_risk_scores)
    || !isAudioMetrics(value.audio_metrics)
  ) {
    throw new ApiError('The backend returned an invalid statistics payload.')
  }
  return {
    active_students: value.active_students,
    active_alerts: value.active_alerts,
    total_logged_alerts: value.total_logged_alerts,
    unique_students_flagged: value.unique_students_flagged,
    severity_counts: value.severity_counts,
    status_counts: value.status_counts,
    category_counts: value.category_counts,
    student_risk_scores: value.student_risk_scores,
    audio_metrics: value.audio_metrics,
  }
}

function parseIncidents(value: unknown): Incident[] {
  if (!Array.isArray(value) || !value.every(isIncident)) {
    throw new ApiError('The backend returned an invalid incident payload.')
  }
  return value
}

function parseConfig(value: unknown): DetectionConfig {
  if (!isRecord(value)
    || typeof value.yaw_threshold !== 'number'
    || !Number.isFinite(value.yaw_threshold)
    || typeof value.pitch_threshold !== 'number'
    || !Number.isFinite(value.pitch_threshold)
    || typeof value.suspicious_duration !== 'number'
    || !Number.isFinite(value.suspicious_duration)
    || typeof value.audio_threshold !== 'number'
    || !Number.isFinite(value.audio_threshold)
    || typeof value.enable_object_detection !== 'boolean'
    || typeof value.enable_absence_detection !== 'boolean'
    || typeof value.enable_audio_detection !== 'boolean') {
    throw new ApiError('The backend returned an invalid configuration payload.')
  }
  return {
    yaw_threshold: value.yaw_threshold,
    pitch_threshold: value.pitch_threshold,
    suspicious_duration: value.suspicious_duration,
    audio_threshold: value.audio_threshold,
    enable_object_detection: value.enable_object_detection,
    enable_absence_detection: value.enable_absence_detection,
    enable_audio_detection: value.enable_audio_detection,
  }
}

async function authorizedResponse(path: string, options?: RequestInit): Promise<Response> {
  if (!apiBaseUrl) {
    throw new ApiError('VITE_API_BASE_URL is not configured. Set it in the frontend deployment environment.')
  }

  let response: Response
  try {
    const headers = new Headers(options?.headers)
    if (options?.body && !headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json')
    }
    if (apiAccessKey) {
      headers.set('X-API-Key', apiAccessKey)
    }
    response = await fetch(`${apiBaseUrl}${path}`, {
      ...options,
      headers,
    })
  } catch {
    throw new ApiError('Could not reach the monitoring backend. Check the network and backend CORS configuration.')
  }

  if (!response.ok) {
    let detail = response.status === 401
      ? 'A valid prototype API access key is required.'
      : response.status === 403
        ? 'The backend denied this request.'
        : `Request failed (${response.status})`
    try {
      const body = await response.json() as { error?: string; message?: string }
      detail = body.error || body.message || detail
    } catch {
      // Some endpoints return HTML rather than JSON.
    }
    throw new ApiError(detail, response.status)
  }

  return response
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await authorizedResponse(path, options)
  try {
    return await response.json() as T
  } catch {
    throw new ApiError('The backend returned an invalid response.')
  }
}

async function requestBlob(path: string): Promise<Blob> {
  return (await authorizedResponse(path)).blob()
}

async function streamEvents(signal: AbortSignal, onMessage: (data: string) => void): Promise<void> {
  const response = await authorizedResponse('/api/stream-events', { signal })
  if (!response.body) {
    throw new ApiError('The backend event stream did not provide a response body.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) return
      buffer += decoder.decode(value, { stream: true })
      const messages = buffer.split(/\r?\n\r?\n/)
      buffer = messages.pop() ?? ''
      for (const message of messages) {
        const data = message
          .split(/\r?\n/)
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart())
          .join('\n')
        if (data) onMessage(data)
      }
    }
  } finally {
    reader.releaseLock()
  }
}

export const api = {
  stats: async () => parseStats(await request<unknown>('/api/stats')),
  alerts: async () => parseIncidents(await request<unknown>('/api/alerts')),
  streamEvents,
  exportIncidents: () => requestBlob('/api/incidents/export'),
  printableReport: (examName: string) => {
    const query = new URLSearchParams()
    if (examName.trim()) query.set('exam_name', examName.trim())
    const queryString = query.toString()
    return requestBlob(`/api/report/pdf${queryString ? `?${queryString}` : ''}`)
  },
  evidence: (filename: string) => requestBlob(`/screenshots/${encodeURIComponent(filename)}`),
  updateIncidentStatus: (id: number, status: 'CONFIRMED' | 'DISMISSED') =>
    request<{ status: string; incident_id: number; new_status: string }>(`/api/incidents/${id}/status`, {
      method: 'POST',
      body: JSON.stringify({ status }),
    }),
  config: async () => parseConfig(await request<unknown>('/api/config')),
  saveConfig: (config: DetectionConfig) =>
    request<unknown>('/api/config', {
      method: 'POST',
      body: JSON.stringify({
        yaw: config.yaw_threshold,
        pitch: config.pitch_threshold,
        duration: config.suspicious_duration,
        audio_threshold: config.audio_threshold,
        enable_object_detection: config.enable_object_detection,
        enable_absence_detection: config.enable_absence_detection,
        enable_audio_detection: config.enable_audio_detection,
      }),
    }).then(parseConfig),
  control: (action: 'pause' | 'resume') =>
    request<{ status: string }>(`/api/control/${action}`, { method: 'POST' }),
}
