import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, AlertTriangle, Bell, BookOpen, Check, ChevronDown,
  ChevronRight, CircleHelp, ClipboardList, Clock3, Download, Eye, FileBarChart2, Fingerprint,
  Gauge, LayoutDashboard, LogOut, Menu, Mic, Monitor, MoreHorizontal, Pause, Play, Radio,
  Search, Settings, Shield, ShieldAlert, ShieldCheck, Siren, Smartphone, Users, Video,
  X, Crosshair, ScanFace, Waves,
} from 'lucide-react'
import {
  Area, AreaChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { Link, NavLink, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { ApiError, api, clearApiAccessKey, getApiBaseUrl, hasApiAccessKey, setApiAccessKey, type DashboardStats, type DetectionConfig, type Incident } from './api'
import './App.css'

type BackendState = 'loading' | 'connected' | 'error'
interface DashboardContextValue {
  stats: DashboardStats | null
  incidents: Incident[]
  backendState: BackendState
  backendError: string
  eventStreamState: 'connecting' | 'connected' | 'disconnected'
  lastUpdated: Date | null
  refresh: () => Promise<void>
  updateIncident: (id: number, status: 'CONFIRMED' | 'DISMISSED') => Promise<void>
}

const DashboardContext = createContext<DashboardContextValue | null>(null)

const navigation = [
  { label: 'Overview', path: '/', icon: LayoutDashboard },
  { label: 'Live monitoring', path: '/monitoring', icon: Video },
  { label: 'Students', path: '/students', icon: Users },
  { label: 'Suspicious activities', path: '/activities', icon: ShieldAlert },
  { label: 'Risk analysis', path: '/risk', icon: Gauge },
  { label: 'Reports', path: '/reports', icon: FileBarChart2 },
]

const initialStats = (stats: DashboardStats | null) => [
  { label: 'Students present', value: stats?.active_students, icon: Users, tone: 'blue', note: 'Active student detections' },
  { label: 'Live alerts', value: stats?.active_alerts, icon: Siren, tone: 'amber', note: 'In the current frame' },
  { label: 'Logged incidents', value: stats?.total_logged_alerts, icon: ClipboardList, tone: 'violet', note: 'Recorded by detection system' },
  { label: 'Students flagged', value: stats?.unique_students_flagged, icon: ShieldAlert, tone: 'red', note: 'With an incident record' },
]

function useDashboard() {
  const context = useContext(DashboardContext)
  if (!context) throw new Error('Dashboard context is unavailable')
  return context
}

function App() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [backendState, setBackendState] = useState<BackendState>('loading')
  const [backendError, setBackendError] = useState('')
  const [eventStreamState, setEventStreamState] = useState<DashboardContextValue['eventStreamState']>(() => getApiBaseUrl() ? 'connecting' : 'disconnected')
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [accessKeyRevision, setAccessKeyRevision] = useState(0)
  const refresh = useCallback(async () => {
    setBackendState((state) => state === 'connected' ? state : 'loading')
    const [statsResult, incidentsResult] = await Promise.allSettled([api.stats(), api.alerts()])
    if (statsResult.status === 'fulfilled' && incidentsResult.status === 'fulfilled') {
      setStats(statsResult.value)
      setIncidents(incidentsResult.value)
      setBackendState('connected')
      setBackendError('')
      setLastUpdated(new Date())
      return
    }
    const rejected = statsResult.status === 'rejected' ? statsResult.reason : incidentsResult.status === 'rejected' ? incidentsResult.reason : null
    setBackendState('error')
    setBackendError(rejected instanceof Error ? rejected.message : 'The backend returned an unexpected response.')
  }, [])
  const updateIncident = useCallback(async (id: number, status: 'CONFIRMED' | 'DISMISSED') => {
    await api.updateIncidentStatus(id, status)
    await refresh()
  }, [refresh])

  useEffect(() => {
    const handleAccessKeyChange = () => setAccessKeyRevision((revision) => revision + 1)
    window.addEventListener('api-access-key-change', handleAccessKeyChange)
    return () => window.removeEventListener('api-access-key-change', handleAccessKeyChange)
  }, [])

  useEffect(() => {
    const initialLoad = window.setTimeout(() => void refresh(), 0)
    const timer = window.setInterval(() => void refresh(), 30000)
    const baseUrl = getApiBaseUrl()
    if (!baseUrl) {
      return () => {
        window.clearTimeout(initialLoad)
        window.clearInterval(timer)
      }
    }

    const controller = new AbortController()
    let retryTimer: number | undefined
    const connect = () => {
      if (controller.signal.aborted) return
      setEventStreamState('connecting')
      void api.streamEvents(controller.signal, (data) => {
        try {
        const telemetry = JSON.parse(data) as Partial<DashboardStats> & { new_alert?: Incident | null }
        setStats((current) => current ? {
          ...current,
          active_students: telemetry.active_students ?? current.active_students,
          active_alerts: telemetry.active_alerts ?? current.active_alerts,
          total_logged_alerts: telemetry.total_logged_alerts ?? current.total_logged_alerts,
          student_risk_scores: telemetry.student_risk_scores ?? current.student_risk_scores,
          audio_metrics: telemetry.audio_metrics ?? current.audio_metrics,
        } : current)
        setLastUpdated(new Date())
        if (telemetry.new_alert) void refresh()
      } catch {
        setEventStreamState('disconnected')
      }
      }).then(() => {
      if (!controller.signal.aborted) {
        setEventStreamState('disconnected')
        retryTimer = window.setTimeout(connect, 3000)
      }
      }).catch((reason: unknown) => {
      if (controller.signal.aborted) return
      setEventStreamState('disconnected')
      if (!(reason instanceof ApiError) || (reason.status !== 401 && reason.status !== 503)) {
        retryTimer = window.setTimeout(connect, 3000)
      }
      })
    }
    connect()

    return () => {
      window.clearTimeout(initialLoad)
      window.clearInterval(timer)
      window.clearTimeout(retryTimer)
      controller.abort()
    }
  }, [refresh, accessKeyRevision])

  const value = useMemo(() => ({
    stats, incidents, backendState, backendError, eventStreamState, lastUpdated, refresh, updateIncident,
  }), [stats, incidents, backendState, backendError, eventStreamState, lastUpdated, refresh, updateIncident])

  return (
    <DashboardContext.Provider value={value}>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="*" element={<DashboardShell />} />
      </Routes>
    </DashboardContext.Provider>
  )
}

function DashboardShell() {
  const { stats, incidents, backendState, backendError, eventStreamState, lastUpdated, refresh } = useDashboard()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [search, setSearch] = useState('')
  const location = useLocation()
  const navigate = useNavigate()
  const currentPage = navigation.find((item) => item.path === location.pathname)?.label ?? pageTitle(location.pathname)
  const notificationCount = stats?.active_alerts ?? 0

  const submitSearch = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    navigate(`/students?q=${encodeURIComponent(search.trim())}`)
  }

  return (
    <div className="app-shell">
      {mobileMenuOpen && <button className="sidebar-backdrop" aria-label="Close navigation" onClick={() => setMobileMenuOpen(false)} />}
      <aside className={`sidebar ${mobileMenuOpen ? 'sidebar-open' : ''}`}>
        <Link className="brand" to="/" aria-label="Holistic Detection overview">
          <span className="brand-mark"><Crosshair size={22} strokeWidth={2.2} /></span>
          <span><strong>holistic</strong><small>DETECTION</small></span>
        </Link>
        <div className="workspace-label">EXAM OPERATIONS</div>
        <nav className="side-nav" aria-label="Primary navigation">
          {navigation.map(({ label, path, icon: Icon }) => (
            <NavLink key={path} to={path} end={path === '/'} onClick={() => setMobileMenuOpen(false)} className={({ isActive }) => `nav-link ${isActive ? 'nav-active' : ''}`}>
              <Icon size={18} /><span>{label}</span>
              {label === 'Suspicious activities' && notificationCount > 0 && <span className="nav-count">{notificationCount}</span>}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="connection-card">
            <span className={`connection-dot ${backendState === 'connected' ? 'dot-live' : backendState === 'error' ? 'dot-error' : ''}`} />
            <div><strong>{backendState === 'connected' ? 'Backend connected' : backendState === 'error' ? 'Backend unavailable' : 'Connecting…'}</strong><small>{eventStreamState === 'connected' ? 'Live events connected' : 'Live events disconnected'}</small></div>
          </div>
          <NavLink to="/settings" onClick={() => setMobileMenuOpen(false)} className={({ isActive }) => `nav-link ${isActive ? 'nav-active' : ''}`}><Settings size={18} /><span>Settings</span></NavLink>
          <Link className="nav-link" onClick={() => setMobileMenuOpen(false)} to="/login"><LogOut size={18} /><span>Sign in / sign out</span></Link>
          <div className="operator-card">
            <div className="avatar">PO</div><div><strong>Proctor console</strong><small>Session not authenticated</small></div><MoreHorizontal size={17} />
          </div>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setMobileMenuOpen((open) => !open)} aria-label="Open navigation"><Menu size={20} /></button>
          <div className="breadcrumbs"><span>Examination</span><ChevronRight size={14} /><strong>{currentPage}</strong></div>
          <div className="topbar-actions">
            <div className="exam-status"><span className={`connection-dot ${backendState === 'connected' ? 'dot-live' : backendState === 'error' ? 'dot-error' : ''}`} />{backendState === 'connected' ? 'Backend connected' : backendState === 'error' ? 'Connection issue' : 'Connecting'}</div>
            <time className="local-time" dateTime={new Date().toISOString()}>{new Date().toLocaleString([], { weekday: 'short', hour: 'numeric', minute: '2-digit' })}</time>
            <form className="search-box" onSubmit={submitSearch}><Search size={16} /><input aria-label="Search students" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search students…" /><kbd>↵</kbd></form>
            <Link to="/activities" className="icon-button notification-button" aria-label={`${notificationCount} active alerts`}><Bell size={19} />{notificationCount > 0 && <span className="notification-dot" />}</Link>
            <div className="top-profile"><div className="avatar avatar-small">PO</div><span>Proctor</span><ChevronDown size={15} /></div>
          </div>
        </header>
        {backendState === 'error' && <div className="backend-banner" role="status"><AlertTriangle size={17} /><span><strong>Backend data unavailable.</strong> {backendError} Student, alert, and analytics data will appear when the API is reachable.</span><button onClick={() => void refresh()}>Retry</button></div>}
        <main className="page-content">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/monitoring" element={<MonitoringPage />} />
            <Route path="/students" element={<StudentsPage />} />
            <Route path="/activities" element={<ActivitiesPage />} />
            <Route path="/risk" element={<RiskPage />} />
            <Route path="/reports" element={<ReportsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
          <footer className="page-footer"><span>Holistic Detection <span className="footer-separator">·</span> Examination monitoring console</span><span>{lastUpdated ? `Last synced ${lastUpdated.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : 'Waiting for backend data'}</span></footer>
        </main>
      </div>
      <span className="visually-hidden" aria-live="polite">{incidents.length} incident records loaded</span>
    </div>
  )
}

function PageHeading({ eyebrow, title, description, children }: { eyebrow: string; title: string; description: string; children?: React.ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{children && <div className="heading-actions">{children}</div>}</div>
}

function Panel({ title, subtitle, action, children, className = '' }: { title: string; subtitle?: string; action?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return <section className={`panel ${className}`}><div className="panel-heading"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>{children}</section>
}

function OverviewPage() {
  const { stats, incidents, backendState, eventStreamState } = useDashboard()
  const highRisk = Object.values(stats?.student_risk_scores ?? {}).filter((score) => Number(score) > 70).length
  const riskData = riskDistribution(stats?.student_risk_scores ?? {})
  const recent = incidents.slice(0, 5)
  return (
    <>
      <PageHeading eyebrow="MONITORING OVERVIEW" title="Examination overview" description="A clear view of examination activity and system health." />
      <div className="session-strip">
        <span className="session-icon"><BookOpen size={19} /></span>
        <div className="session-copy"><strong>Examination session</strong><span>Session details are not provided by the connected backend.</span></div>
        <div className={`session-status ${backendState === 'connected' ? 'status-connected' : backendState === 'error' ? 'status-error' : ''}`}><span className="connection-dot" />{backendState === 'connected' ? 'Backend online' : backendState === 'error' ? 'Backend offline' : 'Connecting'}</div>
        <span className="session-divider" />
        <div className="session-status"><Radio size={15} />{eventStreamState === 'connected' ? 'Live updates on' : 'Live updates unavailable'}</div>
      </div>
      <div className="stats-grid">
        {initialStats(stats).map(({ label, value, icon: Icon, tone, note }) => (
          <div className="stat-card" key={label}>
            <div className="stat-top"><span className={`stat-icon icon-${tone}`}><Icon size={19} /></span><span className="stat-note">{note}</span></div>
            <div className="stat-number">{value === undefined ? '—' : value.toLocaleString()}</div>
            <div className="stat-label">{label}</div>
          </div>
        ))}
        <div className="stat-card risk-stat">
          <div className="stat-top"><span className="stat-icon icon-red"><ShieldAlert size={19} /></span><span className="risk-chip">{stats ? `${highRisk} high risk` : 'Awaiting data'}</span></div>
          <div className="stat-number">{stats ? highRisk.toLocaleString() : '—'}</div>
          <div className="stat-label">High-risk students <span className="stat-sub-label">Risk score above 70%</span></div>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span className={`stat-icon ${backendState === 'connected' ? 'icon-green' : 'icon-blue'}`}><Monitor size={19} /></span><span className={`system-pill ${backendState === 'connected' ? 'pill-live' : ''}`}>{backendState === 'connected' ? 'Connected' : backendState === 'error' ? 'Unavailable' : 'Checking'}</span></div>
          <div className="stat-number stat-status-word">{backendState === 'connected' ? 'Online' : backendState === 'error' ? 'Offline' : '—'}</div>
          <div className="stat-label">Monitoring backend <span className="stat-sub-label">Session state not exposed</span></div>
        </div>
      </div>

      <div className="overview-grid">
        <Panel title="Risk distribution" subtitle="Student telemetry from the backend" action={<Link className="text-link" to="/risk">View analysis <ChevronRight size={15} /></Link>}>
          <div className="chart-area risk-chart">
            {riskData.total ? <>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart><Pie data={riskData.data} dataKey="value" nameKey="name" innerRadius={69} outerRadius={96} paddingAngle={4} stroke="none">
                  {riskData.data.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
                </Pie><Tooltip formatter={(value) => [`${value} students`, '']} /></PieChart>
              </ResponsiveContainer>
              <div className="chart-center"><strong>{riskData.total}</strong><span>students</span></div>
            </> : <EmptyState icon={<Gauge size={21} />} title="No risk scores available" body={backendState === 'connected' ? 'Risk distribution will appear when the backend reports student scores.' : 'Connect to the backend to load student risk telemetry.'} />}
          </div>
          <div className="legend-row">{riskData.data.map((entry) => <div className="legend-item" key={entry.name}><span style={{ backgroundColor: entry.color }} />{entry.name}<strong>{entry.value}</strong></div>)}</div>
        </Panel>
        <Panel title="Activity summary" subtitle="Incident records grouped by detection category" action={<Link className="text-link" to="/activities">All activity <ChevronRight size={15} /></Link>}>
          <CategorySummary incidents={incidents} />
        </Panel>
      </div>

      <Panel title="Recent suspicious activity" subtitle="Latest events recorded by the detection backend" action={<Link className="text-link" to="/activities">View all events <ChevronRight size={15} /></Link>} className="recent-panel">
        <IncidentTable incidents={recent} compact />
      </Panel>
    </>
  )
}

function CategorySummary({ incidents }: { incidents: Incident[] }) {
  const categories = useMemo(() => {
    const counts = new Map<string, number>()
    incidents.forEach((incident) => {
      const key = incident.category || 'Uncategorized'
      counts.set(key, (counts.get(key) ?? 0) + 1)
    })
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5)
  }, [incidents])
  const icons = [Eye, Smartphone, Waves, Activity, Fingerprint]
  if (categories.length === 0) return <EmptyState icon={<Activity size={21} />} title="No activity to summarize" body="Category counts will be based on incident records returned by the backend." />
  const max = categories[0]?.[1] ?? 1
  return <div className="category-list">{categories.map(([name, count], index) => {
    const Icon = icons[index % icons.length]
    return <div className="category-row" key={name}><span className="category-icon"><Icon size={16} /></span><div className="category-info"><div><strong>{name.replaceAll('_', ' ')}</strong><span>{count}</span></div><div className="progress-track"><span style={{ width: `${Math.max(4, count / max * 100)}%` }} /></div></div></div>
  })}</div>
}

function MonitoringPage() {
  const { stats, incidents, backendState } = useDashboard()
  const [cameraError, setCameraError] = useState('')
  const [cameraActive, setCameraActive] = useState(false)
  const [cameraStarting, setCameraStarting] = useState(false)
  const [serverMonitor, setServerMonitor] = useState<'unknown' | 'running' | 'paused'>('unknown')
  const [controlError, setControlError] = useState('')
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const mountedRef = useRef(true)
  const highRiskCount = Object.values(stats?.student_risk_scores ?? {}).filter((score) => Number(score) > 70).length
  const startCamera = async () => {
    if (cameraStarting || cameraActive) return
    setCameraStarting(true)
    setCameraError('')
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera access is not available in this browser or page context. Use HTTPS and a supported browser.')
      setCameraStarting(false)
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop())
        return
      }
      streamRef.current = stream
      if (videoRef.current) videoRef.current.srcObject = stream
      setCameraActive(true)
    } catch (error) {
      setCameraError(error instanceof DOMException && error.name === 'NotAllowedError'
        ? 'Camera permission was denied. Allow camera access in your browser settings and try again.'
        : error instanceof DOMException && error.name === 'NotFoundError'
          ? 'No camera was found on this device.'
          : 'The browser could not start the camera. Check device access and try again.')
    } finally {
      if (mountedRef.current) setCameraStarting(false)
    }
  }
  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraActive(false)
  }
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])
  const changeServerMonitoring = async (action: 'pause' | 'resume') => {
    setControlError('')
    try {
      const response = await api.control(action)
      setServerMonitor(response.status === 'paused' ? 'paused' : response.status === 'running' ? 'running' : 'unknown')
    } catch (error) {
      setControlError(error instanceof Error ? error.message : 'Could not change backend monitoring state.')
    }
  }
  const recent = incidents.slice(0, 4)
  return <>
    <PageHeading eyebrow="PROCTORING CONSOLE" title="Live monitoring" description="Browser camera preview and backend-reported examination telemetry.">
      <span className="live-tag"><span className="connection-dot dot-live" />{backendState === 'connected' ? 'Backend connected' : 'Backend status unavailable'}</span>
    </PageHeading>
    <div className="monitoring-layout">
      <div className="monitor-main">
        <Panel title="Local camera preview" subtitle="This camera stays in your browser; no video is uploaded." action={<span className={`local-camera-status ${cameraActive ? 'is-on' : ''}`}><span className="connection-dot" />{cameraActive ? 'Camera active' : 'Camera off'}</span>} className="video-panel">
          <div className={`camera-preview ${cameraActive ? 'camera-on' : ''}`}>
            <video ref={videoRef} autoPlay muted playsInline aria-label="Local camera preview" />
            {!cameraActive && <div className="camera-idle"><span><Video size={28} /></span><strong>Camera preview is off</strong><p>Start the camera to view this device locally.</p><button className="button button-primary" onClick={() => void startCamera()} disabled={cameraStarting}><Video size={16} />{cameraStarting ? 'Starting…' : 'Enable camera'}</button></div>}
            {cameraActive && <div className="camera-overlay"><span><span className="connection-dot dot-live" />LOCAL PREVIEW</span><span>No AI analysis</span></div>}
          </div>
          {cameraError && <p className="inline-error" role="alert">{cameraError}</p>}
          {cameraActive && <div className="camera-controls"><span><Shield size={15} />Local preview only — not sent to the server</span><button className="button button-secondary" onClick={stopCamera}><X size={15} />Stop camera</button></div>}
        </Panel>
        <div className="module-grid">
          <DetectionModule icon={<Smartphone size={17} />} title="Object detection" detail="Phone · paper · prohibited objects" />
          <DetectionModule icon={<Activity size={17} />} title="Pose estimation" detail="Head and body movement" />
          <DetectionModule icon={<Eye size={17} />} title="Gaze analysis" detail="Looking away and attention" />
          <DetectionModule icon={<ScanFace size={17} />} title="Face matching" detail="Student identity matching" />
          <DetectionModule icon={<Mic size={17} />} title="Audio monitoring" detail="Audio anomaly detection" />
        </div>
        <p className="module-disclaimer"><CircleHelp size={15} /> Detection confidence and per-module status are not included in the backend API response.</p>
      </div>
      <aside className="monitor-side">
        <Panel title="Monitoring controls" subtitle="Controls the backend detection process">
          <div className="server-state"><span className={`server-state-icon ${serverMonitor === 'running' ? 'server-running' : serverMonitor === 'paused' ? 'server-paused' : ''}`}><Monitor size={19} /></span><div><strong>{serverMonitor === 'unknown' ? 'State not reported' : serverMonitor === 'running' ? 'Monitoring resumed' : 'Monitoring paused'}</strong><small>The API does not expose its current state.</small></div></div>
          {controlError && <p className="inline-error" role="alert">{controlError}</p>}
          <div className="control-actions"><button className="button button-secondary" onClick={() => void changeServerMonitoring('pause')}><Pause size={15} />Pause</button><button className="button button-primary" onClick={() => void changeServerMonitoring('resume')}><Play size={15} />Resume</button></div>
        </Panel>
        <div className="mini-metric-grid"><MiniMetric title="Active students" value={stats?.active_students} icon={<Users size={16} />} /><MiniMetric title="Active alerts" value={stats?.active_alerts} icon={<Siren size={16} />} /><MiniMetric title="High risk" value={stats ? highRiskCount : undefined} icon={<ShieldAlert size={16} />} /><MiniMetric title="Incident records" value={stats?.total_logged_alerts} icon={<ClipboardList size={16} />} /></div>
        <Panel title="Latest alerts" subtitle="From the incident log" action={<Link className="text-link" to="/activities">View all</Link>}><RecentIncidentList incidents={recent} /></Panel>
      </aside>
    </div>
  </>
}

function DetectionModule({ icon, title, detail }: { icon: React.ReactNode; title: string; detail: string }) {
  return <div className="module-card"><span className="module-icon">{icon}</span><div><strong>{title}</strong><small>{detail}</small></div><span className="module-unavailable">Not reported</span></div>
}

function MiniMetric({ title, value, icon }: { title: string; value?: number; icon: React.ReactNode }) {
  return <div className="mini-metric"><span>{icon}</span><strong>{value === undefined ? '—' : value}</strong><small>{title}</small></div>
}

function StudentsPage() {
  const { stats, incidents, backendState } = useDashboard()
  const location = useLocation()
  const params = new URLSearchParams(location.search)
  const query = params.get('q')?.toLowerCase() ?? ''
  const [riskFilter, setRiskFilter] = useState('all')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const studentRows = useMemo(() => {
    const records = new Map<string, { score?: number; incidents: Incident[] }>()
    Object.entries(stats?.student_risk_scores ?? {}).forEach(([id, score]) => records.set(String(id), { score: Number(score), incidents: [] }))
    incidents.forEach((incident) => {
      const id = String(incident.student_id)
      const existing = records.get(id) ?? { incidents: [] }
      existing.incidents.push(incident)
      if (existing.score === undefined && typeof incident.risk_score === 'number') existing.score = incident.risk_score
      records.set(id, existing)
    })
    return [...records.entries()].map(([id, record]) => ({ id, ...record })).sort((a, b) => (b.score ?? -1) - (a.score ?? -1))
  }, [stats, incidents])
  const filtered = studentRows.filter((student) => {
    const score = student.score
    const risk = score === undefined ? 'unknown' : score > 70 ? 'high' : score > 30 ? 'medium' : 'low'
    return (!query || student.id.toLowerCase().includes(query)) && (riskFilter === 'all' || risk === riskFilter)
  })
  const selected = studentRows.find((student) => student.id === selectedId)
  return <>
    <PageHeading eyebrow="EXAM CANDIDATES" title="Students" description="Student identities and risk details available from the detection telemetry." />
    <div className="data-notice"><CircleHelp size={16} /><span>Student name, roll number, hall, and seat fields are not exposed by the current backend. Listed students are identified only by backend student ID.</span></div>
    <div className="table-toolbar"><div className="table-summary"><strong>{backendState === 'connected' ? studentRows.length : '—'}</strong> student IDs in risk or incident data</div><label className="filter-control"><span>Risk level</span><select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)}><option value="all">All levels</option><option value="low">Low (0–30%)</option><option value="medium">Medium (31–70%)</option><option value="high">High (&gt;70%)</option></select></label></div>
    <Panel title="Student monitoring records" subtitle="Derived from risk scores and logged incidents">
      {filtered.length > 0 ? <div className="table-wrap"><table><thead><tr><th>Student ID</th><th>Risk score</th><th>Risk level</th><th>Incidents</th><th>Latest recorded activity</th><th /></tr></thead><tbody>{filtered.map((student) => <tr key={student.id}><td><button className="student-link" onClick={() => setSelectedId(student.id)}><span className="student-avatar">{student.id.slice(-2)}</span><strong>Student {student.id}</strong></button></td><td>{student.score === undefined ? 'Not reported' : `${Math.round(student.score)}%`}</td><td><RiskBadge score={student.score} /></td><td>{student.incidents.length}</td><td>{student.incidents[0] ? formatDate(student.incidents[0].timestamp) : 'No incident record'}</td><td><button className="icon-button table-action" aria-label={`View Student ${student.id}`} onClick={() => setSelectedId(student.id)}><ChevronRight size={17} /></button></td></tr>)}</tbody></table></div>
        : <EmptyState icon={<Users size={21} />} title={backendState === 'connected' ? 'No student records available' : 'Student data unavailable'} body={backendState === 'connected' ? 'The backend has not reported any student risk scores or incident records.' : 'Connect to the backend to load student telemetry.'} />}
    </Panel>
    {selected && <StudentDetails id={selected.id} score={selected.score} incidents={selected.incidents} onClose={() => setSelectedId(null)} />}
  </>
}

function StudentDetails({ id, score, incidents, onClose }: { id: string; score?: number; incidents: Incident[]; onClose: () => void }) {
  return <div className="modal-backdrop" role="presentation" onClick={onClose}><section className="detail-modal" role="dialog" aria-modal="true" aria-labelledby="student-details-title" onClick={(event) => event.stopPropagation()}><div className="modal-heading"><div><div className="eyebrow">STUDENT RECORD</div><h2 id="student-details-title">Student {id}</h2></div><button className="icon-button" aria-label="Close details" onClick={onClose}><X size={18} /></button></div><div className="detail-stats"><div><small>Risk score</small><strong>{score === undefined ? 'Not reported' : `${Math.round(score)}%`}</strong></div><div><small>Logged incidents</small><strong>{incidents.length}</strong></div><div><small>Identity details</small><strong>Not available</strong></div></div><h3>Monitoring history</h3>{incidents.length ? <RecentIncidentList incidents={incidents.slice(0, 8)} /> : <p className="muted-copy">No incident history returned for this student.</p>}</section></div>
}

function ActivitiesPage() {
  const { incidents, backendState } = useDashboard()
  const [riskFilter, setRiskFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState('all')
  const [query, setQuery] = useState('')
  const [actionError, setActionError] = useState('')
  const [updating, setUpdating] = useState<number | null>(null)
  const { updateIncident } = useDashboard()
  const filtered = incidents.filter((incident) => {
    const score = incident.risk_score
    const risk = typeof score !== 'number' ? 'unknown' : score > 70 ? 'high' : score > 30 ? 'medium' : 'low'
    const matchesRisk = riskFilter === 'all' || risk === riskFilter
    const matchesStatus = statusFilter === 'all' || (incident.status || 'UNREVIEWED').toLowerCase() === statusFilter
    const haystack = `${incident.id} ${incident.student_id} ${incident.activity} ${incident.category}`.toLowerCase()
    return matchesRisk && matchesStatus && haystack.includes(query.toLowerCase())
  })
  const setStatus = async (id: number, status: 'CONFIRMED' | 'DISMISSED') => {
    setUpdating(id)
    setActionError('')
    try {
      await updateIncident(id, status)
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Could not update incident review status.')
    } finally {
      setUpdating(null)
    }
  }
  return <>
    <PageHeading eyebrow="INCIDENT REVIEW" title="Suspicious activities" description="Review backend-recorded incidents and update their review status." />
    <div className="activity-summary-row"><SummaryTile label="All incidents" value={backendState === 'connected' ? incidents.length : undefined} icon={<ClipboardList size={17} />} /><SummaryTile label="Unreviewed" value={backendState === 'connected' ? incidents.filter((item) => !item.status || item.status === 'UNREVIEWED').length : undefined} icon={<Clock3 size={17} />} /><SummaryTile label="High risk" value={backendState === 'connected' ? incidents.filter((item) => (item.risk_score ?? 0) > 70).length : undefined} icon={<ShieldAlert size={17} />} /></div>
    <div className="filter-bar"><label className="search-box table-search"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search incidents…" /></label><label className="filter-control"><span>Risk</span><select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)}><option value="all">All risk levels</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></label><label className="filter-control"><span>Review status</span><select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}><option value="all">All statuses</option><option value="unreviewed">Unreviewed</option><option value="confirmed">Confirmed</option><option value="dismissed">Dismissed</option></select></label></div>
    {actionError && <div className="inline-error" role="alert">{actionError}</div>}
    <Panel title="Incident log" subtitle={`${filtered.length} of ${backendState === 'connected' ? incidents.length : '—'} records`}>
      {filtered.length ? <IncidentTable incidents={filtered} onReview={setStatus} updating={updating} /> : <EmptyState icon={<ShieldCheck size={21} />} title={backendState === 'connected' ? 'No matching incidents' : 'Incident data unavailable'} body={backendState === 'connected' ? incidents.length ? 'Try adjusting the filters or search query.' : 'The backend has not recorded suspicious activities.' : 'Connect to the backend to load incident records.'} />}
    </Panel>
  </>
}

function SummaryTile({ label, value, icon }: { label: string; value?: number; icon: React.ReactNode }) {
  return <div className="summary-tile"><span>{icon}</span><div><strong>{value === undefined ? '—' : value.toLocaleString()}</strong><small>{label}</small></div></div>
}

function RiskPage() {
  const { stats, incidents, backendState } = useDashboard()
  const riskData = riskDistribution(stats?.student_risk_scores ?? {})
  const highRisk = Object.entries(stats?.student_risk_scores ?? {}).filter(([, score]) => Number(score) > 70).sort((a, b) => Number(b[1]) - Number(a[1]))
  const trend = useMemo(() => {
    const days = new Map<string, number>()
    incidents.forEach(({ timestamp }) => {
      const date = new Date(timestamp)
      if (Number.isNaN(date.getTime())) return
      const label = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
      days.set(label, (days.get(label) ?? 0) + 1)
    })
    return [...days.entries()].map(([date, count]) => ({ date, count })).slice(-10)
  }, [incidents])
  const hourlyActivity = useMemo(() => {
    const counts = new Map<number, number>()
    incidents.forEach(({ timestamp }) => {
      const date = new Date(timestamp)
      if (Number.isNaN(date.getTime())) return
      counts.set(date.getHours(), (counts.get(date.getHours()) ?? 0) + 1)
    })
    return [...counts.entries()].sort(([hourA], [hourB]) => hourA - hourB).map(([hour, count]) => ({
      hour: new Date(2000, 0, 1, hour).toLocaleTimeString([], { hour: 'numeric' }),
      count,
    }))
  }, [incidents])
  const categories = useMemo(() => {
    const values = new Map<string, number>()
    incidents.forEach(({ category }) => {
      const key = category || 'Uncategorized'
      values.set(key, (values.get(key) ?? 0) + 1)
    })
    return [...values.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6)
  }, [incidents])
  return <>
    <PageHeading eyebrow="ANALYTICS" title="Risk analysis" description="Risk distribution and activity trends derived from live backend records." />
    <div className="risk-overview"><div className="risk-overview-icon"><Gauge size={20} /></div><div><small>Students with a reported risk score</small><strong>{riskData.total || (backendState === 'connected' ? '0' : '—')}</strong></div><div className="risk-level-brief"><span className="risk-dot dot-high" />{riskData.data[2]?.value ?? 0} high risk<span className="brief-divider" /><span className="risk-dot dot-medium" />{riskData.data[1]?.value ?? 0} medium<span className="brief-divider" /><span className="risk-dot dot-low" />{riskData.data[0]?.value ?? 0} low</div></div>
    <div className="analysis-grid">
      <Panel title="Incident trend" subtitle="Recorded incidents by date" className="trend-panel">
        {trend.length ? <div className="trend-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={trend} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}><defs><linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#2785e8" stopOpacity={0.18} /><stop offset="95%" stopColor="#2785e8" stopOpacity={0.01} /></linearGradient></defs><CartesianGrid stroke="#edf1f6" vertical={false} /><XAxis dataKey="date" tickLine={false} axisLine={false} tick={{ fill: '#8a94a5', fontSize: 11 }} /><YAxis allowDecimals={false} tickLine={false} axisLine={false} tick={{ fill: '#8a94a5', fontSize: 11 }} /><Tooltip /><Area type="monotone" dataKey="count" name="Incidents" stroke="#2785e8" strokeWidth={2.3} fill="url(#trendFill)" /></AreaChart></ResponsiveContainer></div> : <EmptyState icon={<Activity size={21} />} title="No incident trend available" body="Trend data will appear after timestamped incidents are reported." />}
      </Panel>
      <Panel title="Risk categories" subtitle="Current student risk scores">
        {riskData.total ? <div className="risk-breakdown">{riskData.data.map((entry) => <div className="risk-break-row" key={entry.name}><div className="risk-break-label"><span style={{ backgroundColor: entry.color }} />{entry.name}<strong>{entry.value}</strong></div><div className="progress-track"><span style={{ width: `${entry.value / riskData.total * 100}%`, backgroundColor: entry.color }} /></div></div>)}</div> : <EmptyState icon={<Gauge size={21} />} title="No risk scores reported" body="Student risk scores are not available yet." />}
      </Panel>
    </div>
    <div className="analysis-grid lower-analysis">
      <Panel title="Highest reported risk" subtitle="Students above 70% risk">
        {highRisk.length ? <div className="high-risk-list">{highRisk.slice(0, 8).map(([id, score], index) => <div className="high-risk-row" key={id}><span className="risk-rank">{index + 1}</span><span className="student-avatar">{id.slice(-2)}</span><strong>Student {id}</strong><span className="risk-meter"><span style={{ width: `${Math.min(100, Math.max(0, Number(score)))}%` }} /></span><RiskBadge score={Number(score)} /></div>)}</div> : <EmptyState icon={<ShieldCheck size={21} />} title="No high-risk students" body={backendState === 'connected' ? 'No reported student scores currently exceed 70%.' : 'Risk data is not available.'} />}
      </Panel>
      <Panel title="Detection categories" subtitle="Incident counts returned by the backend">
        {categories.length ? <div className="category-list">{categories.map(([name, count]) => <div className="category-row" key={name}><span className="category-icon"><Activity size={16} /></span><div className="category-info"><div><strong>{name}</strong><span>{count}</span></div><div className="progress-track"><span style={{ width: `${count / (categories[0]?.[1] ?? 1) * 100}%` }} /></div></div></div>)}</div> : <EmptyState icon={<Activity size={21} />} title="No category data" body="Module performance metrics are not included in the API." />}
      </Panel>
      <Panel title="Hourly activity" subtitle="Incident counts by local hour">
        {hourlyActivity.length ? <div className="trend-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={hourlyActivity} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}><CartesianGrid stroke="#edf1f6" vertical={false} /><XAxis dataKey="hour" tickLine={false} axisLine={false} tick={{ fill: '#8a94a5', fontSize: 10 }} /><YAxis allowDecimals={false} tickLine={false} axisLine={false} tick={{ fill: '#8a94a5', fontSize: 10 }} /><Tooltip /><Area type="monotone" dataKey="count" name="Incidents" stroke="#7186cb" strokeWidth={2.2} fill="#eef0fb" /></AreaChart></ResponsiveContainer></div> : <EmptyState icon={<Clock3 size={21} />} title="No hourly activity" body="The chart appears when the backend returns timestamped incident records." />}
      </Panel>
    </div>
  </>
}

function ReportsPage() {
  const { stats, incidents, backendState } = useDashboard()
  const [examName, setExamName] = useState('')
  const [exporting, setExporting] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const exportCsv = async () => {
    setExporting(true)
    setError('')
    setMessage('')
    try {
      const blob = await api.exportIncidents()
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = 'proctoring_incident_report.csv'
      document.body.append(anchor)
      anchor.click()
      anchor.remove()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
      setMessage('Incident CSV downloaded.')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'The CSV report could not be downloaded.')
    } finally {
      setExporting(false)
    }
  }
  const openPrintableReport = async () => {
    setError('')
    const reportWindow = window.open('about:blank', '_blank')
    if (!reportWindow) {
      setError('Allow pop-ups to open the printable report.')
      return
    }
    reportWindow.opener = null
    try {
      const report = await api.printableReport(examName)
      const url = URL.createObjectURL(report)
      reportWindow.location.href = url
      window.setTimeout(() => URL.revokeObjectURL(url), 60000)
    } catch (reason) {
      reportWindow.close()
      setError(reason instanceof Error ? reason.message : 'The printable report could not be opened.')
    }
  }
  return <>
    <PageHeading eyebrow="REPORTING" title="Reports" description="Export incident data or open the backend-generated examination summary." />
    <div className="report-notice"><FileBarChart2 size={18} /><div><strong>Available reports</strong><span>CSV incident export and an HTML report intended for printing. A PDF file is not generated by the current API.</span></div></div>
    {error && <div className="inline-error" role="alert">{error}</div>}{message && <div className="inline-success" role="status"><Check size={16} />{message}</div>}
    <div className="report-grid">
      <article className="report-card"><span className="report-icon report-blue"><ClipboardList size={21} /></span><div className="eyebrow">INCIDENT DATA</div><h2>Suspicious activity export</h2><p>Download the recorded incident log as a CSV file for review or archiving.</p><div className="report-detail"><span>Included records</span><strong>{backendState === 'connected' ? incidents.length.toLocaleString() : 'Unavailable'}</strong></div><button className="button button-primary" onClick={() => void exportCsv()} disabled={exporting || backendState !== 'connected'}><Download size={16} />{exporting ? 'Preparing…' : 'Download CSV'}</button></article>
      <article className="report-card"><span className="report-icon report-violet"><FileBarChart2 size={21} /></span><div className="eyebrow">SESSION SUMMARY</div><h2>Printable examination report</h2><p>Open the backend-generated printable HTML summary. Use your browser’s print dialog to save as PDF.</p><label className="exam-name-field">Examination name <input value={examName} onChange={(event) => setExamName(event.target.value)} placeholder="Use backend default" /></label><button className="button button-primary" onClick={openPrintableReport}><FileBarChart2 size={16} />Open printable report</button></article>
    </div>
    <div className="report-stats">{[['Students present', stats?.active_students], ['Logged incidents', stats?.total_logged_alerts], ['Flagged students', stats?.unique_students_flagged]].map(([label, value]) => <div key={String(label)}><span>{label}</span><strong>{typeof value === 'number' ? value.toLocaleString() : '—'}</strong></div>)}</div>
    <Panel title="Activity history" subtitle="Recent backend incident records">
      <IncidentTable incidents={incidents.slice(0, 10)} compact />
    </Panel>
  </>
}

function SettingsPage() {
  const [config, setConfig] = useState<DetectionConfig | null>(null)
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading')
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)
  const [operatorKey, setOperatorKey] = useState('')
  const [accessKeyReady, setAccessKeyReady] = useState(hasApiAccessKey)
  const [accessKeyMessage, setAccessKeyMessage] = useState('')
  const load = useCallback(async () => {
    setState('loading')
    setError('')
    try {
      setConfig(await api.config())
      setState('ready')
    } catch (reason) {
      setState('error')
      setError(reason instanceof Error ? reason.message : 'Configuration could not be loaded.')
    }
  }, [])
  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0)
    return () => window.clearTimeout(timer)
  }, [load])
  const setNumber = (key: 'yaw_threshold' | 'pitch_threshold' | 'suspicious_duration' | 'audio_threshold', value: string) => {
    setConfig((current) => current ? { ...current, [key]: Number(value) } : current)
    setSaved(false)
  }
  const setToggle = (key: 'enable_object_detection' | 'enable_absence_detection' | 'enable_audio_detection', value: boolean) => {
    setConfig((current) => current ? { ...current, [key]: value } : current)
    setSaved(false)
  }
  const save = async () => {
    if (!config) return
    setError('')
    setSaved(false)
    try {
      setConfig(await api.saveConfig(config))
      setSaved(true)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Settings could not be saved.')
    }
  }
  const enableWriteAccess = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!operatorKey.trim()) {
      setAccessKeyMessage('Enter the prototype API access key provided by the service administrator.')
      return
    }
    setApiAccessKey(operatorKey)
    setOperatorKey('')
    setAccessKeyReady(true)
    setAccessKeyMessage('Key is held in memory for this page session and is checked on protected data and write requests.')
    void load()
  }
  const disableAccessKey = () => {
    clearApiAccessKey()
    setAccessKeyReady(false)
    setAccessKeyMessage('Prototype API access key cleared from this page session.')
  }
  return <>
    <PageHeading eyebrow="SYSTEM PREFERENCES" title="Settings" description="View and update detection configuration supported by the Flask API." />
    <Panel title="Prototype API access restriction" subtitle="Protected data reads and write operations require the API_WRITE_TOKEN configured by the service administrator." className="write-access-panel">
      <form className="write-access-form" onSubmit={enableWriteAccess}>
        <label className="token-field">Shared API access key<input type="password" autoComplete="off" value={operatorKey} onChange={(event) => setOperatorKey(event.target.value)} placeholder="Enter key for this page session" /></label>
        <button className="button button-primary" type="submit"><Shield size={15} />Enable for this session</button>
        {accessKeyReady && <button className="button button-secondary" type="button" onClick={disableAccessKey}>Clear key</button>}
      </form>
      <p className={accessKeyMessage && !accessKeyReady ? 'inline-error' : 'write-access-note'} role={accessKeyMessage ? 'status' : undefined}>
        {accessKeyMessage || 'Temporary prototype restriction only: this shared key is visible to users who enter it and is not user authentication, identity, or role-based authorization. Never use it as an institutional security boundary.'}
      </p>
    </Panel>
    {state === 'loading' && <div className="loading-state"><span className="spinner" />Loading backend configuration…</div>}
    {state === 'error' && <div className="backend-banner" role="alert"><AlertTriangle size={17} /><span>{error}</span><button onClick={() => void load()}>Retry</button></div>}
    {config && <Panel title="Detection thresholds" subtitle="Changes are sent directly to the backend detection configuration">
      <div className="settings-grid">
        <NumberSetting label="Head yaw threshold" description="Horizontal head turn threshold in degrees." value={config.yaw_threshold} onChange={(value) => setNumber('yaw_threshold', value)} />
        <NumberSetting label="Head pitch threshold" description="Vertical head movement threshold in degrees." value={config.pitch_threshold} onChange={(value) => setNumber('pitch_threshold', value)} />
        <NumberSetting label="Suspicious duration" description="Duration threshold used by suspicious activity detection." value={config.suspicious_duration} onChange={(value) => setNumber('suspicious_duration', value)} />
        <NumberSetting label="Audio threshold" description="Noise threshold for audio detection." value={config.audio_threshold} onChange={(value) => setNumber('audio_threshold', value)} />
      </div>
      <div className="toggle-settings"><ToggleSetting label="Object detection" detail="Enable prohibited-object detection." value={config.enable_object_detection} onChange={(value) => setToggle('enable_object_detection', value)} /><ToggleSetting label="Absence detection" detail="Enable student absence checks." value={config.enable_absence_detection} onChange={(value) => setToggle('enable_absence_detection', value)} /><ToggleSetting label="Audio detection" detail="Enable audio anomaly detection." value={config.enable_audio_detection} onChange={(value) => setToggle('enable_audio_detection', value)} /></div>
      {error && <p className="inline-error" role="alert">{error}</p>}{saved && <p className="inline-success" role="status"><Check size={16} />Backend settings saved.</p>}
      <div className="settings-footer"><span>Only settings exposed by the backend can be changed here.</span><button className="button button-primary" onClick={() => void save()} disabled={state !== 'ready' || !accessKeyReady}><Check size={15} />Save settings</button></div>
    </Panel>}
    <div className="data-notice"><Shield size={16} /><span>User accounts, role-based access, and session management are not implemented. The API key is shared, browser-visible to anyone who enters it, and only a temporary prototype restriction.</span></div>
  </>
}

function NumberSetting({ label, description, value, onChange }: { label: string; description: string; value: number; onChange: (value: string) => void }) {
  return <label className="number-setting"><span><strong>{label}</strong><small>{description}</small></span><input type="number" step="any" value={value} onChange={(event) => onChange(event.target.value)} /></label>
}

function ToggleSetting({ label, detail, value, onChange }: { label: string; detail: string; value: boolean; onChange: (value: boolean) => void }) {
  return <label className="toggle-setting"><span><strong>{label}</strong><small>{detail}</small></span><input type="checkbox" checked={value} onChange={(event) => onChange(event.target.checked)} /><span className="toggle-visual" /></label>
}

function LoginPage() {
  return <div className="login-page"><Link className="brand login-brand" to="/"><span className="brand-mark"><Crosshair size={22} /></span><span><strong>holistic</strong><small>DETECTION</small></span></Link><section className="login-card"><span className="login-icon"><Shield size={21} /></span><div className="eyebrow">PROCTOR ACCESS</div><h1>User sign-in unavailable</h1><p>The backend does not provide user accounts or session authentication. No credentials are collected by this page.</p><div className="login-footnote"><Shield size={14} />The API key is a temporary shared prototype restriction, not user authentication or institutional-grade security.</div><Link className="text-link login-back" to="/settings">Open API access settings <ChevronRight size={15} /></Link><Link className="text-link login-back" to="/">Return to dashboard <ChevronRight size={15} /></Link></section><span className="login-footer">Holistic Detection · Examination monitoring</span></div>
}

function IncidentTable({ incidents, compact = false, onReview, updating }: { incidents: Incident[]; compact?: boolean; onReview?: (id: number, status: 'CONFIRMED' | 'DISMISSED') => void; updating?: number | null }) {
  if (!incidents.length) return <EmptyState icon={<ClipboardList size={21} />} title="No incident records" body="Incident details will appear here when the backend reports them." />
  return <div className="table-wrap"><table><thead><tr><th>Event</th><th>Student ID</th><th>Activity</th><th>Detection</th><th>Timestamp</th><th>Risk</th><th>Status</th>{!compact && <th>Review</th>}</tr></thead><tbody>{incidents.map((incident) => <tr key={incident.id}><td><span className="event-id">#{incident.id}</span></td><td><span className="table-student">Student {incident.student_id}</span></td><td className="activity-cell">{incident.activity || '—'}{incident.screenshot_path && <EvidenceLink path={incident.screenshot_path} />}</td><td><span className="category-chip">{(incident.category || 'Uncategorized').replaceAll('_', ' ')}</span></td><td className="time-cell">{formatDate(incident.timestamp)}</td><td><RiskBadge score={typeof incident.risk_score === 'number' ? incident.risk_score : undefined} /></td><td><StatusBadge status={incident.status} /></td>{!compact && <td><div className="review-actions">{incident.status === 'UNREVIEWED' || !incident.status ? <><button className="review-button review-confirm" title="Confirm incident" aria-label={`Confirm event ${incident.id}`} disabled={updating === incident.id} onClick={() => onReview?.(incident.id, 'CONFIRMED')}><Check size={15} /></button><button className="review-button review-dismiss" title="Dismiss incident" aria-label={`Dismiss event ${incident.id}`} disabled={updating === incident.id} onClick={() => onReview?.(incident.id, 'DISMISSED')}><X size={15} /></button></> : <span className="reviewed-label">Reviewed</span>}</div></td>}</tr>)}</tbody></table></div>
}

function EvidenceLink({ path }: { path: string }) {
  const [error, setError] = useState('')
  const file = path.split(/[\\/]/).filter(Boolean).pop()
  if (!file) return null
  const openEvidence = async () => {
    setError('')
    const evidenceWindow = window.open('about:blank', '_blank')
    if (!evidenceWindow) {
      setError('Allow pop-ups to open evidence.')
      return
    }
    evidenceWindow.opener = null
    try {
      const evidence = await api.evidence(file)
      const url = URL.createObjectURL(evidence)
      evidenceWindow.location.href = url
      window.setTimeout(() => URL.revokeObjectURL(url), 60000)
    } catch (reason) {
      evidenceWindow.close()
      setError(reason instanceof Error ? reason.message : 'Evidence could not be opened.')
    }
  }
  return <><button className="evidence-link" type="button" onClick={() => void openEvidence()}><Eye size={13} />Evidence</button>{error && <span className="evidence-error" role="alert">{error}</span>}</>
}

function RecentIncidentList({ incidents }: { incidents: Incident[] }) {
  if (!incidents.length) return <div className="small-empty">No incidents reported.</div>
  return <div className="recent-list">{incidents.map((incident) => <div className="recent-item" key={incident.id}><span className={`recent-indicator ${riskClass(incident.risk_score)}`}><AlertTriangle size={14} /></span><div className="recent-copy"><strong>{incident.activity || 'Suspicious activity'}</strong><span>Student {incident.student_id} <span>·</span> {formatRelativeDate(incident.timestamp)}</span></div><RiskBadge score={typeof incident.risk_score === 'number' ? incident.risk_score : undefined} /></div>)}</div>
}

function RiskBadge({ score }: { score?: number }) {
  if (score === undefined || !Number.isFinite(score)) return <span className="risk-badge risk-unknown">Not scored</span>
  const level = score > 70 ? 'high' : score > 30 ? 'medium' : 'low'
  return <span className={`risk-badge risk-${level}`}><span />{level} · {Math.round(score)}%</span>
}

function StatusBadge({ status }: { status?: string | null }) {
  const value = status || 'UNREVIEWED'
  return <span className={`status-badge status-${value.toLowerCase()}`}>{value.replaceAll('_', ' ')}</span>
}

function EmptyState({ icon, title, body }: { icon: React.ReactNode; title: string; body: string }) {
  return <div className="empty-state"><span className="empty-icon">{icon}</span><strong>{title}</strong><p>{body}</p></div>
}

function NotFoundPage() {
  return <div className="not-found"><span className="empty-icon"><CircleHelp size={22} /></span><h1>Page not found</h1><p>The page you requested does not exist in this console.</p><Link className="button button-primary" to="/">Back to overview</Link></div>
}

function riskDistribution(scores: Record<string, number>) {
  const values = Object.values(scores).map(Number).filter(Number.isFinite)
  const counts = { Low: 0, Medium: 0, High: 0 }
  values.forEach((score) => {
    if (score > 70) counts.High += 1
    else if (score > 30) counts.Medium += 1
    else counts.Low += 1
  })
  return {
    total: values.length,
    data: [
      { name: 'Low', value: counts.Low, color: '#34a879' },
      { name: 'Medium', value: counts.Medium, color: '#e5a331' },
      { name: 'High', value: counts.High, color: '#e26464' },
    ],
  }
}

function riskClass(score?: number | null) {
  if (score === undefined || score === null) return 'indicator-neutral'
  return score > 70 ? 'indicator-high' : score > 30 ? 'indicator-medium' : 'indicator-low'
}

function formatDate(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value || '—' : date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

function formatRelativeDate(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value || 'Unknown time'
  const minutes = Math.round((date.getTime() - Date.now()) / 60000)
  if (Math.abs(minutes) < 1) return 'Just now'
  if (Math.abs(minutes) < 60) return `${Math.abs(minutes)}m ${minutes < 0 ? 'ago' : 'from now'}`
  const hours = Math.round(minutes / 60)
  if (Math.abs(hours) < 24) return `${Math.abs(hours)}h ${hours < 0 ? 'ago' : 'from now'}`
  return date.toLocaleDateString()
}

function pageTitle(path: string) {
  return navigation.find((item) => item.path === path)?.label ?? (path === '/settings' ? 'Settings' : 'Dashboard')
}

export default App
