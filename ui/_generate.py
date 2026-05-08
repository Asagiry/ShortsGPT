"""Generate all React/TS source files for the desktop app."""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.join(BASE, "src", "components")
os.makedirs(COMP, exist_ok=True)


def w(rel: str, content: str):
    p = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  wrote {rel}")


# ── App.tsx ──────────────────────────────────────────────────────────
w("src/App.tsx", r"""import React, { useState, useEffect, useRef, useCallback } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import LogPanel from './components/LogPanel';

export type JobEvent = { type: 'started' | 'log' | 'done'; text?: string; code?: number; error?: string };
export type AppPhase = 'idle' | 'running' | 'done' | 'error';
export const STEPS = ['Resolve', 'Transcribe', 'Find', 'Verify', 'Render'] as const;

export default function App() {
  const [source, setSource] = useState('');
  const [outputDir, setOutputDir] = useState('');
  const [clips, setClips] = useState('auto');
  const [quality, setQuality] = useState('1080');
  const [whisper, setWhisper] = useState('base');
  const [language, setLanguage] = useState('auto');
  const [frame, setFrame] = useState('fit');
  const [phase, setPhase] = useState<AppPhase>('idle');
  const [statusText, setStatusText] = useState('Ready');
  const [progress, setProgress] = useState(0);
  const [activeStep, setActiveStep] = useState(-1);
  const [logs, setLogs] = useState<string[]>([]);
  const [errorMsg, setErrorMsg] = useState('');
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => { logEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs]);

  useEffect(() => {
    window.__onStudioEvent = (ev: JobEvent) => {
      if (ev.type === 'started') { setPhase('running'); setStatusText('Running'); setProgress(0); setActiveStep(0); setErrorMsg(''); setLogs([]); }
      else if (ev.type === 'log' && ev.text) { setLogs(p => [...p, ev.text!]); updateProgress(ev.text); }
      else if (ev.type === 'done') {
        if (ev.code === 0) { setPhase('done'); setStatusText('Completed'); setProgress(100); setActiveStep(STEPS.length); }
        else { setPhase('error'); setStatusText(ev.error || `Stopped (${ev.code})`); setErrorMsg(ev.error || `Exit code ${ev.code}`); }
      }
    };
  }, []);

  const updateProgress = useCallback((line: string) => {
    if (line.includes('[transcribe]') && line.includes('%')) {
      try { const p = parseFloat(line.split('%')[0].split(/\s+/).pop() || '0'); setProgress(v => Math.max(v, Math.min(35, p * 0.35))); setStatusText(`Transcribing ${p.toFixed(1)}%`); setActiveStep(1); } catch {}
    } else if (line.includes('Finding highlight')) { setProgress(45); setStatusText('Finding highlights'); setActiveStep(2); }
    else if (line.includes('Verifying best moments')) { setProgress(55); setStatusText('Verifying best moments'); setActiveStep(3); }
    else if (line.includes('Rendering shorts')) { setProgress(65); setStatusText('Rendering shorts'); setActiveStep(4); }
    else if (line.includes('[clip/local] short')) {
      try { const m = line.match(/short\s+(\d+)\/(\d+)/); if (m) { const c = +m[1], t = +m[2]; setProgress(65 + ((c-1)/Math.max(t,1))*30); setStatusText(`Rendering short ${c}/${t}`); } } catch {}
    }
  }, []);

  const isRunning = phase === 'running';

  return (
    <div className="app-shell">
      <Header phase={phase} statusText={statusText} />
      <div className="body">
        <Sidebar source={source} setSource={setSource} outputDir={outputDir} setOutputDir={setOutputDir}
          clips={clips} setClips={setClips} quality={quality} setQuality={setQuality}
          whisper={whisper} setWhisper={setWhisper} language={language} setLanguage={setLanguage}
          frame={frame} setFrame={setFrame} isRunning={isRunning}
          onChooseFile={async()=>{const p=await window.pywebview?.api?.choose_file();if(p)setSource(p);}}
          onChooseDir={async()=>{const p=await window.pywebview?.api?.choose_directory();if(p)setOutputDir(p);}}
          onPaste={async()=>{try{const t=await navigator.clipboard.readText();if(t?.trim())setSource(t.trim());}catch{}}}
          onStart={async()=>{if(!source.trim())return;const r=await window.pywebview?.api?.start_job({source:source.trim(),output_dir:outputDir||`${source.trim()}_output`,clips,quality,whisper,language,frame});if(r&&!r.ok){setPhase('error');setErrorMsg(r.error||'Failed');}}}
          onStop={async()=>{await window.pywebview?.api?.stop_job();}}
          onOpenOutput={()=>{if(outputDir)window.pywebview?.api?.open_directory(outputDir);}}
          onClear={()=>{setLogs([]);setProgress(0);setPhase('idle');setStatusText('Ready');setActiveStep(-1);setErrorMsg('');}}
        />
        <LogPanel phase={phase} isRunning={isRunning} progress={progress} activeStep={activeStep} errorMsg={errorMsg} logs={logs} logEndRef={logEndRef} />
      </div>
    </div>
  );
}
""")

# ── Header.tsx ───────────────────────────────────────────────────────
w("src/components/Header.tsx", r"""import React from 'react';
import { Monitor, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';
import { AppPhase } from '../App';

export default function Header({ phase, statusText }: { phase: AppPhase; statusText: string }) {
  const cls: Record<AppPhase, string> = { idle: 'badge-idle', running: 'badge-running', done: 'badge-done', error: 'badge-error' };
  const ico: Record<AppPhase, React.ReactNode> = {
    idle: <CheckCircle2 size={14}/>, running: <Loader2 size={14} className="spin"/>,
    done: <CheckCircle2 size={14}/>, error: <AlertCircle size={14}/>,
  };
  return (
    <header className="hero">
      <div className="hero-left">
        <div className="hero-icon"><Monitor size={22}/></div>
        <div>
          <h1 className="hero-title">AI Shorts Studio</h1>
          <p className="hero-sub">Any video to viral moments to rendered shorts. Resume-safe, live logs, no CMD.</p>
        </div>
      </div>
      <div className={'badge ' + cls[phase]}>{ico[phase]} {statusText}</div>
    </header>
  );
}
""")

# ── Sidebar.tsx ──────────────────────────────────────────────────────
w("src/components/Sidebar.tsx", r"""import React from 'react';
import { Play, Square, FolderOpen, FileVideo, Clipboard, Trash2, Zap, Languages, Film, Hash, FolderOutput, Monitor } from 'lucide-react';

type Props = {
  source: string; setSource: (v: string) => void;
  outputDir: string; setOutputDir: (v: string) => void;
  clips: string; setClips: (v: string) => void;
  quality: string; setQuality: (v: string) => void;
  whisper: string; setWhisper: (v: string) => void;
  language: string; setLanguage: (v: string) => void;
  frame: string; setFrame: (v: string) => void;
  isRunning: boolean;
  onChooseFile: () => void; onChooseDir: () => void; onPaste: () => void;
  onStart: () => void; onStop: () => void; onOpenOutput: () => void; onClear: () => void;
};

function Combo({ label, icon, value, onChange, options, disabled }: {
  label: string; icon: React.ReactNode; value: string; onChange: (v: string) => void; options: string[]; disabled?: boolean;
}) {
  return (
    <div className="combo">
      <span className="combo-label">{icon} {label}</span>
      <select className="combo-select" value={value} onChange={e => onChange(e.target.value)} disabled={disabled}>
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
}

export default function Sidebar(p: Props) {
  return (
    <aside className="sidebar">
      <div className="section">
        <div className="section-head"><span className="section-num">1.</span><span className="section-title">Source</span></div>
        <div className="input-row">
          <input className="field" placeholder="Video URL (YouTube, TikTok, Instagram, Twitter...) or local file..." value={p.source} onChange={e => p.setSource(e.target.value)} disabled={p.isRunning} />
        </div>
        <div className="btn-row">
          <button className="btn btn-ghost" onClick={p.onChooseFile} disabled={p.isRunning}><FileVideo size={15}/> Choose file</button>
          <button className="btn btn-ghost" onClick={p.onPaste} disabled={p.isRunning}><Clipboard size={15}/> Paste</button>
        </div>
      </div>
      <div className="divider" />
      <div className="section">
        <div className="section-head"><span className="section-num">2.</span><span className="section-title">Profile</span></div>
        <div className="grid-2">
          <Combo label="Clips" icon={<Hash size={13}/>} value={p.clips} onChange={p.setClips} options={['auto','1','2','3','5','8','10','12','16']} disabled={p.isRunning} />
          <Combo label="Quality" icon={<Monitor size={13}/>} value={p.quality} onChange={p.setQuality} options={['360','480','720','1080']} disabled={p.isRunning} />
          <Combo label="Whisper" icon={<Zap size={13}/>} value={p.whisper} onChange={p.setWhisper} options={['tiny','base','small','medium']} disabled={p.isRunning} />
          <Combo label="Language" icon={<Languages size={13}/>} value={p.language} onChange={p.setLanguage} options={['auto','ru','en']} disabled={p.isRunning} />
        </div>
        <div style={{marginTop:10}}>
          <Combo label="Frame mode" icon={<Film size={13}/>} value={p.frame} onChange={p.setFrame} options={['fit','crop','face']} disabled={p.isRunning} />
        </div>
      </div>
      <div className="divider" />
      <div className="section">
        <div className="section-head"><span className="section-num">3.</span><span className="section-title">Output</span></div>
        <div className="input-row">
          <input className="field field-sm" placeholder="Output folder..." value={p.outputDir} onChange={e => p.setOutputDir(e.target.value)} disabled={p.isRunning} />
          <button className="btn btn-ghost btn-icon" onClick={p.onChooseDir} disabled={p.isRunning}><FolderOutput size={15}/></button>
        </div>
      </div>
      <div className="divider" />
      <div className="action-block">
        {!p.isRunning ? (
          <button className="btn btn-primary btn-lg" onClick={p.onStart} disabled={!p.source.trim()}><Play size={17}/> Generate shorts</button>
        ) : (
          <button className="btn btn-danger btn-lg" onClick={p.onStop}><Square size={17}/> Stop</button>
        )}
        <div className="btn-row" style={{marginTop:8}}>
          <button className="btn btn-ghost btn-sm" onClick={p.onOpenOutput}><FolderOpen size={14}/> Open output</button>
          <button className="btn btn-ghost btn-sm" onClick={p.onClear}><Trash2 size={14}/> Clear</button>
        </div>
      </div>
    </aside>
  );
}
""")

# ── LogPanel.tsx ─────────────────────────────────────────────────────
w("src/components/LogPanel.tsx", r"""import React, { RefObject } from 'react';
import { Activity, CheckCircle2, AlertCircle, Loader2, XCircle } from 'lucide-react';
import { AppPhase, STEPS } from '../App';

type Props = {
  phase: AppPhase; isRunning: boolean; progress: number; activeStep: number;
  errorMsg: string; logs: string[]; logEndRef: RefObject<HTMLDivElement | null>;
};

export default function LogPanel({ phase, isRunning, progress, activeStep, errorMsg, logs, logEndRef }: Props) {
  return (
    <main className="main-panel">
      <div className="stepper">
        {STEPS.map((label, i) => (
          <div key={label} className={'step' + (i < activeStep ? ' done' : i === activeStep ? ' active' : '')}>
            <div className="step-dot">
              {i < activeStep ? <CheckCircle2 size={14}/> :
               i === activeStep && isRunning ? <Loader2 size={14} className="spin"/> :
               i === activeStep && phase === 'error' ? <XCircle size={14}/> :
               <span className="step-num">{i+1}</span>}
            </div>
            <span className="step-label">{label}</span>
          </div>
        ))}
      </div>
      <div className="progress-bar"><div className="progress-fill" style={{width: progress+'%'}}/></div>
      {errorMsg && <div className="error-banner"><AlertCircle size={16}/><span>{errorMsg}</span></div>}
      <div className="log-panel">
        <div className="log-header"><Activity size={15}/><span>Pipeline log</span><span className="log-count">{logs.length} lines</span></div>
        <div className="log-body">
          {logs.length === 0 && phase === 'idle' && (
            <div className="log-empty"><Activity size={32} strokeWidth={1}/><p>Log output will appear here when you start a job.</p><p className="dim">No text files. Everything is live in-app.</p></div>
          )}
          {logs.map((line, i) => <div key={i} className="log-line">{line}</div>)}
          <div ref={logEndRef}/>
        </div>
      </div>
    </main>
  );
}
""")

# ── index.css ────────────────────────────────────────────────────────
w("src/index.css", r"""

:root {
  --bg: #070a12; --panel: #0d1220; --card: #111827; --border: #1f2937;
  --border-focus: #3b82f6; --text: #e5e7eb; --text-dim: #94a3b8; --text-muted: #64748b;
  --primary: #3b82f6; --primary-hover: #2563eb; --danger: #ef4444; --danger-hover: #dc2626;
  --success: #22c55e; --teal: #0f766e;
  --radius: 10px; --radius-sm: 6px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body { font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); overflow: hidden; }

.app-shell { display: flex; flex-direction: column; height: 100vh; width: 100vw; }

/* ── Hero ── */
.hero { display: flex; align-items: center; justify-content: space-between; padding: 20px 28px; background: linear-gradient(135deg, #0d1a33 0%, #101b35 100%); border-bottom: 1px solid var(--border); }
.hero-left { display: flex; align-items: center; gap: 16px; }
.hero-icon { width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); display: flex; align-items: center; justify-content: center; color: #fff; }
.hero-title { font-size: 22px; font-weight: 700; color: #fff; letter-spacing: -0.3px; }
.hero-sub { font-size: 12px; color: var(--text-dim); margin-top: 2px; }

.badge { display: inline-flex; align-items: center; gap: 7px; padding: 7px 16px; border-radius: 20px; font-size: 12px; font-weight: 600; white-space: nowrap; }
.badge-idle { background: #1e293b; color: var(--text-dim); }
.badge-running { background: rgba(59,130,246,0.18); color: #93c5fd; }
.badge-done { background: rgba(34,197,94,0.15); color: #86efac; }
.badge-error { background: rgba(239,68,68,0.15); color: #fca5a5; }

/* ── Body ── */
.body { display: flex; flex: 1; overflow: hidden; }

/* ── Sidebar ── */
.sidebar { width: 380px; min-width: 340px; background: var(--panel); border-right: 1px solid var(--border); padding: 20px; display: flex; flex-direction: column; gap: 0; overflow-y: auto; }

.section { margin-bottom: 4px; }
.section-head { display: flex; align-items: center; gap: 6px; margin-bottom: 10px; }
.section-num { color: var(--primary); font-weight: 700; font-size: 13px; }
.section-title { font-weight: 600; font-size: 14px; color: #fff; }

.divider { height: 1px; background: var(--border); margin: 16px 0; }

.input-row { display: flex; gap: 8px; margin-bottom: 8px; }
.field { flex: 1; background: #060914; border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 10px 12px; color: #f8fafc; font-size: 13px; font-family: 'Inter', sans-serif; outline: none; transition: border-color 0.2s; }
.field:focus { border-color: var(--border-focus); }
.field-sm { font-size: 12px; padding: 8px 10px; }

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }

.combo { }
.combo-label { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--text-dim); margin-bottom: 5px; font-weight: 500; }
.combo-select { width: 100%; background: #060914; border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 8px 10px; color: var(--text); font-size: 12px; font-family: 'Inter', sans-serif; outline: none; cursor: pointer; appearance: none; transition: border-color 0.2s; }
.combo-select:focus { border-color: var(--border-focus); }

.btn-row { display: flex; gap: 8px; }
.btn { display: inline-flex; align-items: center; justify-content: center; gap: 7px; border: none; border-radius: var(--radius-sm); cursor: pointer; font-family: 'Inter', sans-serif; font-weight: 600; font-size: 13px; transition: all 0.15s; outline: none; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-primary { background: var(--primary); color: #fff; }
.btn-primary:hover:not(:disabled) { background: var(--primary-hover); }
.btn-danger { background: var(--danger); color: #fff; }
.btn-danger:hover:not(:disabled) { background: var(--danger-hover); }
.btn-ghost { background: transparent; color: var(--text-dim); border: 1px solid var(--border); }
.btn-ghost:hover:not(:disabled) { background: var(--card); color: var(--text); }
.btn-icon { padding: 8px; min-width: 38px; }
.btn-sm { font-size: 12px; padding: 6px 12px; }
.btn-lg { font-size: 14px; padding: 12px 20px; width: 100%; border-radius: var(--radius); }

.action-block { margin-top: auto; }

/* ── Main panel ── */
.main-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; background: var(--bg); }

.stepper { display: flex; gap: 6px; padding: 16px 20px; background: var(--panel); border-bottom: 1px solid var(--border); }
.step { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 6px; }
.step-dot { width: 30px; height: 30px; border-radius: 50%; background: var(--card); border: 2px solid var(--border); display: flex; align-items: center; justify-content: center; color: var(--text-dim); transition: all 0.3s; }
.step.active .step-dot { border-color: var(--primary); color: var(--primary); box-shadow: 0 0 12px rgba(59,130,246,0.3); }
.step.done .step-dot { border-color: var(--success); color: var(--success); background: rgba(34,197,94,0.1); }
.step-num { font-size: 11px; font-weight: 700; }
.step-label { font-size: 10px; font-weight: 600; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
.step.active .step-label { color: var(--primary); }
.step.done .step-label { color: var(--success); }

.progress-bar { height: 3px; background: var(--border); }
.progress-fill { height: 100%; background: linear-gradient(90deg, var(--primary), var(--success)); transition: width 0.5s ease; border-radius: 0 2px 2px 0; }

.error-banner { display: flex; align-items: center; gap: 10px; margin: 12px 20px; padding: 12px 16px; background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.25); border-radius: var(--radius-sm); color: #fca5a5; font-size: 13px; }

.log-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; margin: 12px 20px 20px; border-radius: var(--radius); border: 1px solid var(--border); background: #030712; }
.log-header { display: flex; align-items: center; gap: 8px; padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 12px; font-weight: 600; color: var(--text-dim); }
.log-count { margin-left: auto; font-weight: 400; color: var(--text-muted); font-size: 11px; }
.log-body { flex: 1; overflow-y: auto; padding: 14px 16px; font-family: 'JetBrains Mono', monospace; font-size: 12px; line-height: 1.7; }
.log-line { color: #d1d5db; white-space: pre-wrap; word-break: break-all; }
.log-line:hover { background: rgba(255,255,255,0.03); }
.log-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); gap: 12px; text-align: center; font-family: 'Inter', sans-serif; font-size: 13px; }
.log-empty .dim { font-size: 11px; color: var(--text-muted); opacity: 0.6; }

/* ── Spinner ── */
.spin { animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
""")

print("\nAll files generated.")
