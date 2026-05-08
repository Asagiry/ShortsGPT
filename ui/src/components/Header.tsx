import React from 'react';
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
