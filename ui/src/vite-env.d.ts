/// <reference types="vite/client" />

type JobEvent = {
  type: 'started' | 'log' | 'done';
  text?: string;
  code?: number;
  error?: string;
};

type PywebviewApi = {
  choose_file: () => Promise<string | null>;
  choose_files: () => Promise<string[]>;
  choose_directory: () => Promise<string | null>;
  open_directory: (path: string) => Promise<void>;
  start_job: (settings: any) => Promise<{ ok: boolean; error?: string }>;
  stop_job: () => Promise<void>;
  load_settings: () => Promise<Record<string, string>>;
  save_settings: (data: Record<string, string>) => Promise<void>;
};

declare global {
  interface Window {
    pywebview: { api: PywebviewApi };
    __onStudioEvent: ((ev: JobEvent) => void) | undefined;
  }
}

export {};
