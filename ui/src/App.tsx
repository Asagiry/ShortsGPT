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
  const [llmFastModel, setLlmFastModel] = useState('');
  const [llmBeatModel, setLlmBeatModel] = useState('');
  const [llmStrongModel, setLlmStrongModel] = useState('');
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
      if (settings.llm_fast_model) setLlmFastModel(settings.llm_fast_model);
      if (settings.llm_beat_model) setLlmBeatModel(settings.llm_beat_model);
      if (settings.llm_strong_model) setLlmStrongModel(settings.llm_strong_model);
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
    if (line.includes('Preparing source') || line.includes('Source ready') || line.includes('Job folder')) {
      setProgress(v => Math.max(v, 3)); setStatusText('Preparing source'); setActiveStep(0);
    } else if (line.includes('Transcript ready') || line.includes('loaded from cache')) {
      setProgress(v => Math.max(v, 35)); setStatusText('Transcript loaded'); setActiveStep(1);
    } else if (line.includes('Media analysis ready') || line.includes('Analyzing video and audio')) {
      setProgress(v => Math.max(v, 37)); setStatusText('Analyzing media'); setActiveStep(2);
    } else if (line.includes('[transcribe]') && line.includes('%')) {
      try { const p = parseFloat(line.split('%')[0].split(/\s+/).pop() || '0'); setProgress(v => Math.max(v, Math.min(35, p * 0.35))); setStatusText(`Transcribing ${p.toFixed(1)}%`); setActiveStep(1); } catch {}
    } else if (line.includes('Choosing edit style') || line.includes('Edit style selected')) { setProgress(v => Math.max(v, 38)); setStatusText('Choosing edit style'); setActiveStep(2); }
    else if (line.includes('Planning clip count') || line.includes('Clip count')) { setProgress(v => Math.max(v, 40)); setStatusText('Planning clip count'); setActiveStep(2); }
    else if (line.includes('GPT story mapping') || line.includes('GPT story map') || line.includes('Story beats')) { setProgress(v => Math.max(v, 44)); setStatusText('Mapping story beats'); setActiveStep(3); }
    else if (line.includes('[LLM story beats]') && line.includes('%')) {
      try { const p = parseFloat(line.split('%')[0].split(/\s+/).pop() || '0'); setProgress(v => Math.max(v, 44 + Math.min(10, p * 0.10))); setStatusText(`Mapping story beats ${p.toFixed(0)}%`); setActiveStep(3); } catch {}
    }
    else if (line.includes('GPT clip selection') || line.includes('GPT clip candidates') || line.includes('Candidates')) { setProgress(v => Math.max(v, 54)); setStatusText('Selecting candidates'); setActiveStep(3); }
    else if (line.includes('[LLM candidates]') && line.includes('%')) {
      try { const p = parseFloat(line.split('%')[0].split(/\s+/).pop() || '0'); setProgress(v => Math.max(v, 54 + Math.min(8, p * 0.08))); setStatusText(`Selecting candidates ${p.toFixed(0)}%`); setActiveStep(3); } catch {}
    }
    else if (line.includes('LLM final review') || line.includes('LLM waiting') || line.includes('Checking clip boundaries') || line.includes('Final boundary check')) { setProgress(v => Math.max(v, 63)); setStatusText('Final LLM review'); setActiveStep(4); }
    else if (line.includes('Rendering shorts') || line.includes('Rendering short')) { setProgress(v => Math.max(v, 68)); setStatusText('Rendering shorts'); setActiveStep(5); }
    else if (line.includes('[clip/local] short') || line.includes('Rendering short')) {
      try { const m = line.match(/short\s+(\d+)\/(\d+)/); if (m) { const c = +m[1], t = +m[2]; setProgress(68 + ((c-1)/Math.max(t,1))*28); setStatusText(`Rendering short ${c}/${t}`); } } catch {}
    }
  }, []);

  const isRunning = phase === 'running';
  const selectedSources = sources.length ? sources : source.trim().split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const saveSettings = async () => {
    await window.pywebview?.api?.save_settings?.({output_dir:outputDir,clips,quality,whisper,language,frame,edit_profile:editProfile,llm_api_key:llmApiKey,llm_base_url:llmBaseUrl,llm_model:llmModel,llm_fast_model:llmFastModel,llm_beat_model:llmBeatModel,llm_strong_model:llmStrongModel});
  };

  useEffect(() => {
    if (!settingsLoadedRef.current || isRunning) return;
    const timer = window.setTimeout(() => { saveSettings(); }, 500);
    return () => window.clearTimeout(timer);
  }, [outputDir, clips, quality, whisper, language, frame, editProfile, llmApiKey, llmBaseUrl, llmModel, llmFastModel, llmBeatModel, llmStrongModel, isRunning]);
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
      llm_fast_model: llmFastModel,
      llm_beat_model: llmBeatModel,
      llm_strong_model: llmStrongModel,
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
            <label className="field-label">Fast model</label>
            <input className="field" placeholder="cheap model for profile/count checks" value={llmFastModel} onChange={e => setLlmFastModel(e.target.value)} disabled={isRunning} />
            <label className="field-label">Beat model</label>
            <input className="field" placeholder="cheap/mid model for story map and candidate batches" value={llmBeatModel} onChange={e => setLlmBeatModel(e.target.value)} disabled={isRunning} />
            <label className="field-label">Strong model</label>
            <input className="field" placeholder="best model for final picks and boundaries" value={llmStrongModel} onChange={e => setLlmStrongModel(e.target.value)} disabled={isRunning} />
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
