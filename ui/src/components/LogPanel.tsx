import React, { RefObject } from 'react';
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
