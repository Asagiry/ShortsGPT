import React, { useState, useEffect, useRef, useCallback } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import LogPanel from './components/LogPanel';

export type JobEvent = { type: 'started' | 'log' | 'done'; text?: string; code?: number; error?: string };
export type AppPhase = 'idle' | 'running' | 'done' | 'error';
export const STEPS = ['Resolve', 'Transcribe', 'Plan', 'Find', 'Verify', 'Render'] as const;

export default function App() {
  const [source, setSource] = useState('');
  const [sources, setSources] = useState<string[]>([]);
  const [outputDir, setOutputDir] = useState('');
  const [clips, setClips] = useState('auto');
  const [quality, setQuality] = useState('1080');
  const [whisper, setWhisper] = useState('base');
  const [language, setLanguage] = useState('auto');
  const [frame, setFrame] = useState('fit');
  const [editProfile, setEditProfile] = useState('auto');
  const [llmApiKey, setLlmApiKey] = useState('');
  const [llmBaseUrl, setLlmBaseUrl] = useState('');
  const [llmModel, setLlmModel] = useState('gpt-4o-mini');
  const [showSettings, setShowSettings] = useState(false);
  const [phase, setPhase] = useState<AppPhase>('idle');
  const [statusText, setStatusText] = useState('Ready');
  const [progress, setProgress] = useState(0);
  const [activeStep, setActiveStep] = useState(-1);
  const [logs, setLogs] = useState<string[]>([]);
  const [errorMsg, setErrorMsg] = useState('');
  const logEndRef = useRef<HTMLDivElement>(null);
  const settingsLoadedRef = useRef(false);

  useEffect(() => { logEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs]);

  useEffect(() => {
    (async () => {
      let settings: Record<string, string> | undefined;
      for (let attempt = 0; attempt < 40; attempt += 1) {
        settings = await window.pywebview?.api?.load_settings?.();
        if (settings) break;
        await new Promise(resolve => window.setTimeout(resolve, 100));
      }
      if (!settings) return;
      if (settings.output_dir) setOutputDir(settings.output_dir);
      if (settings.clips) setClips(settings.clips);
      if (settings.quality) setQuality(settings.quality);
      if (settings.whisper) setWhisper(settings.whisper);
      if (settings.language) setLanguage(settings.language);
      if (settings.frame) setFrame(settings.frame);
      if (settings.edit_profile) setEditProfile(settings.edit_profile);
      if (settings.llm_api_key) setLlmApiKey(settings.llm_api_key);
      if (settings.llm_base_url) setLlmBaseUrl(settings.llm_base_url);
      if (settings.llm_model) setLlmModel(settings.llm_model);
      setShowSettings(!settings.llm_api_key || !settings.llm_model);
      settingsLoadedRef.current = true;
    })();
  }, []);

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
    } else if (line.includes('Planning edit rhythm')) { setProgress(38); setStatusText('Planning edit rhythm'); setActiveStep(2); }
    else if (line.includes('Planning clip count')) { setProgress(40); setStatusText('Planning clip count'); setActiveStep(2); }
    else if (line.includes('Building beat map')) { setProgress(44); setStatusText('Building beat map'); setActiveStep(3); }
    else if (line.includes('Finding highlight')) { setProgress(48); setStatusText('Finding highlights'); setActiveStep(3); }
    else if (line.includes('Verifying best moments')) { setProgress(58); setStatusText('Verifying best moments'); setActiveStep(4); }
    else if (line.includes('Rendering shorts')) { setProgress(68); setStatusText('Rendering shorts'); setActiveStep(5); }
    else if (line.includes('[clip/local] short')) {
      try { const m = line.match(/short\s+(\d+)\/(\d+)/); if (m) { const c = +m[1], t = +m[2]; setProgress(68 + ((c-1)/Math.max(t,1))*28); setStatusText(`Rendering short ${c}/${t}`); } } catch {}
    }
  }, []);

  const isRunning = phase === 'running';
  const selectedSources = sources.length ? sources : source.trim().split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const saveSettings = async () => {
    await window.pywebview?.api?.save_settings?.({output_dir:outputDir,clips,quality,whisper,language,frame,edit_profile:editProfile,llm_api_key:llmApiKey,llm_base_url:llmBaseUrl,llm_model:llmModel});
  };

  useEffect(() => {
    if (!settingsLoadedRef.current || isRunning) return;
    const timer = window.setTimeout(() => { saveSettings(); }, 500);
    return () => window.clearTimeout(timer);
  }, [outputDir, clips, quality, whisper, language, frame, editProfile, llmApiKey, llmBaseUrl, llmModel, isRunning]);
  const startJob = async () => {
    if (!selectedSources.length) return;
    await saveSettings();
    const r = await window.pywebview?.api?.start_job({
      source: source.trim(),
      sources: selectedSources,
      output_dir: outputDir || 'output',
      clips,
      quality,
      whisper,
      language,
      frame,
      edit_profile: editProfile,
      llm_api_key: llmApiKey,
      llm_base_url: llmBaseUrl,
      llm_model: llmModel,
    });
    if (r && !r.ok) { setPhase('error'); setErrorMsg(r.error || 'Failed'); }
  };

  return (
    <div className="app-shell">
      <Header phase={phase} statusText={statusText} />
      {showSettings && (
        <div className="modal-backdrop">
          <div className="settings-modal">
            <div className="modal-head">
              <div>
                <h2>LLM settings</h2>
                <p>OpenAI-compatible API for transcript analysis and clip selection.</p>
              </div>
              <button className="btn btn-ghost btn-sm" onClick={() => setShowSettings(false)}>Close</button>
            </div>
            <label className="field-label">API key</label>
            <input className="field" type="password" placeholder="sk-... or provider key" value={llmApiKey} onChange={e => setLlmApiKey(e.target.value)} disabled={isRunning} />
            <label className="field-label">Endpoint</label>
            <input className="field" placeholder="https://api.openai.com/v1 or compatible endpoint" value={llmBaseUrl} onChange={e => setLlmBaseUrl(e.target.value)} disabled={isRunning} />
            <label className="field-label">Model name</label>
            <input className="field" placeholder="gpt-4o-mini, gpt-4.1-mini, provider/model" value={llmModel} onChange={e => setLlmModel(e.target.value)} disabled={isRunning} />
            <button className="btn btn-primary btn-lg" onClick={async()=>{await saveSettings();setShowSettings(false);}}>Save settings</button>
          </div>
        </div>
      )}
      <div className="body">
        <Sidebar source={source} setSource={(v)=>{setSource(v);setSources([]);}} outputDir={outputDir} setOutputDir={setOutputDir}
          clips={clips} setClips={setClips} quality={quality} setQuality={setQuality}
          whisper={whisper} setWhisper={setWhisper} language={language} setLanguage={setLanguage}
          frame={frame} setFrame={setFrame} editProfile={editProfile} setEditProfile={setEditProfile}
          llmApiKey={llmApiKey} setLlmApiKey={setLlmApiKey}
          llmBaseUrl={llmBaseUrl} setLlmBaseUrl={setLlmBaseUrl} llmModel={llmModel} setLlmModel={setLlmModel}
          sourceCount={selectedSources.length} isRunning={isRunning}
          onChooseFile={async()=>{const p=await window.pywebview?.api?.choose_files();if(p?.length){setSources(p);setSource(p.join('\n'));}}}
          onChooseDir={async()=>{const p=await window.pywebview?.api?.choose_directory();if(p)setOutputDir(p);}}
          onPaste={async()=>{try{const t=await navigator.clipboard.readText();if(t?.trim()){setSources([]);setSource(t.trim());}}catch{}}}
          onStart={startJob}
          onStop={async()=>{await window.pywebview?.api?.stop_job();}}
          onOpenOutput={()=>{if(outputDir)window.pywebview?.api?.open_directory(outputDir);}}
          onClear={()=>{setLogs([]);setProgress(0);setPhase('idle');setStatusText('Ready');setActiveStep(-1);setErrorMsg('');}}
          onOpenSettings={()=>setShowSettings(true)}
        />
        <LogPanel phase={phase} isRunning={isRunning} progress={progress} activeStep={activeStep} errorMsg={errorMsg} logs={logs} logEndRef={logEndRef} />
      </div>
    </div>
  );
}
