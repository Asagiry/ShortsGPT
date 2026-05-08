import React from 'react';
import { Play, Square, FolderOpen, FileVideo, Clipboard, Trash2, Zap, Languages, Film, Hash, FolderOutput, Monitor, KeyRound, Server, Brain, Settings } from 'lucide-react';

type Props = {
  source: string; setSource: (v: string) => void;
  outputDir: string; setOutputDir: (v: string) => void;
  clips: string; setClips: (v: string) => void;
  quality: string; setQuality: (v: string) => void;
  whisper: string; setWhisper: (v: string) => void;
  language: string; setLanguage: (v: string) => void;
  frame: string; setFrame: (v: string) => void;
  editProfile: string; setEditProfile: (v: string) => void;
  llmApiKey: string; setLlmApiKey: (v: string) => void;
  llmBaseUrl: string; setLlmBaseUrl: (v: string) => void;
  llmModel: string; setLlmModel: (v: string) => void;
  sourceCount: number;
  isRunning: boolean;
  onChooseFile: () => void; onChooseDir: () => void; onPaste: () => void;
  onStart: () => void; onStop: () => void; onOpenOutput: () => void; onClear: () => void;
  onOpenSettings: () => void;
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
          <textarea className="field source-area" placeholder="Video URL, local file, or one source per line..." value={p.source} onChange={e => p.setSource(e.target.value)} disabled={p.isRunning} />
        </div>
        <div className="btn-row">
          <button className="btn btn-ghost" onClick={p.onChooseFile} disabled={p.isRunning}><FileVideo size={15}/> Choose files</button>
          <button className="btn btn-ghost" onClick={p.onPaste} disabled={p.isRunning}><Clipboard size={15}/> Paste</button>
          {p.sourceCount > 1 && <span className="source-count">{p.sourceCount} queued</span>}
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
        <div style={{marginTop:10}}>
          <Combo label="Edit profile" icon={<Settings size={13}/>} value={p.editProfile} onChange={p.setEditProfile} options={['auto','talking_head','cartoon_dialogue','movie_scene','gameplay','music_visual']} disabled={p.isRunning} />
        </div>
      </div>
      <div className="divider" />
      <div className="section">
        <div className="section-head"><span className="section-num">3.</span><span className="section-title">LLM</span><button className="mini-link" onClick={p.onOpenSettings} disabled={p.isRunning}><Settings size={12}/> Settings</button></div>
        <div className="grid-1">
          <label className="field-label"><KeyRound size={12}/> API key</label>
          <input className="field field-sm" type="password" placeholder="Provider API key" value={p.llmApiKey} onChange={e => p.setLlmApiKey(e.target.value)} disabled={p.isRunning} />
          <label className="field-label"><Server size={12}/> Endpoint</label>
          <input className="field field-sm" placeholder="OpenAI-compatible /v1 endpoint" value={p.llmBaseUrl} onChange={e => p.setLlmBaseUrl(e.target.value)} disabled={p.isRunning} />
          <label className="field-label"><Brain size={12}/> Model</label>
          <input className="field field-sm" placeholder="gpt-4o-mini" value={p.llmModel} onChange={e => p.setLlmModel(e.target.value)} disabled={p.isRunning} />
        </div>
      </div>
      <div className="divider" />
      <div className="section">
        <div className="section-head"><span className="section-num">4.</span><span className="section-title">Output</span></div>
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
