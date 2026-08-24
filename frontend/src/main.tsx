import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

type Result = {
  modality?: string;
  verdict?: string;
  confidence?: number | null;
  status?: string;
  model?: string;
  evidence?: string[];
  filename?: string;
  media_id?: string;
  error?: string;
};

type HistoryItem = Result & { id: string; time: string };

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const HISTORY_KEY = 'deepguard-history';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch { return []; }
  });

  useEffect(() => {
    fetch(`${API}/api/v1/health`).then(r => setApiOnline(r.ok)).catch(() => setApiOnline(false));
    const timer = window.setInterval(() => {
      fetch(`${API}/api/v1/health`).then(r => setApiOnline(r.ok)).catch(() => setApiOnline(false));
    }, 10000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview); };
  }, [preview]);

  const label = useMemo(() => {
    if (!result?.verdict) return 'Awaiting analysis';
    if (result.verdict === 'likely_real') return 'Likely Real';
    if (result.verdict === 'ai_generated') return 'AI Generated';
    return result.verdict.replaceAll('_', ' ');
  }, [result]);

  function selectFile(next: File | undefined) {
    if (!next) return;
    if (!next.type.startsWith('image/')) {
      setResult({ error: 'Please select an image file.' });
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setFile(next);
    setPreview(URL.createObjectURL(next));
    setResult(null);
  }

  async function analyze() {
    if (!file) return;
    setBusy(true); setResult(null);
    const fd = new FormData(); fd.append('media', file);
    try {
      const response = await fetch(`${API}/api/v1/analyze`, { method: 'POST', body: fd });
      const data = await response.json();
      const next: Result = response.ok ? data : { error: data.detail || 'Analysis failed.' };
      setResult(next);
      if (response.ok) {
        const item = { ...next, id: crypto.randomUUID(), time: new Date().toLocaleString() };
        const updated = [item, ...history].slice(0, 6);
        setHistory(updated);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
      }
    } catch {
      setResult({ error: 'API unavailable. Start FastAPI on http://127.0.0.1:8000.' });
    } finally { setBusy(false); }
  }

  function reset() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview(''); setResult(null);
  }

  const confidence = result?.confidence == null ? null : Math.round(result.confidence * 1000) / 10;
  const verdictClass = result?.verdict === 'likely_real' ? 'real' : result?.verdict === 'ai_generated' ? 'ai' : 'neutral';
  const isReal = result?.verdict === 'likely_real';

  const explanation = result && !result.error ? {
    title: isReal ? 'The image is more consistent with the real class.' : 'The image is more consistent with the AI-generated class.',
    summary: isReal
      ? `DEEP-Guard's EfficientNet-B0 classifier assigned the image to the “likely real” class with a ${confidence}% model confidence score. This means the learned visual patterns in this image were closer to the examples the model learned as real than to its AI-generated examples.`
      : `DEEP-Guard's EfficientNet-B0 classifier assigned the image to the “AI generated” class with a ${confidence}% model confidence score. This means the learned visual patterns in this image were closer to the examples the model learned as AI-generated than to its real-image examples.`,
    confidence: confidence == null
      ? 'No confidence score was returned by the API.'
      : `The ${confidence}% value is the model's score for its selected class. It should not be read as a ${confidence}% guarantee that the image is genuinely real or genuinely AI-generated.`,
    evidence: result.evidence?.length
      ? result.evidence
      : ['The API did not return additional evidence for this analysis.'],
    limitation: 'The current baseline exposes a model-level prediction only. It does not yet provide pixel-level heatmaps, manipulated-region detection, metadata analysis, or a human-readable explanation of which exact visual features caused the decision.',
    next: isReal
      ? 'For higher-stakes verification, compare the result with the image source, metadata, provenance, and other independent evidence.'
      : 'For higher-stakes verification, look for the original source, provenance, metadata, and an independent detector before treating the result as conclusive.'
  } : null;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark">DG</div><div><strong>DEEP-Guard</strong><small>Media Forensics</small></div></div>
        <nav><a className="active">Overview</a><a>Image analysis</a><a>Models</a><a>Evidence</a></nav>
        <div className="sidebar-note">Authenticity assessment, not absolute proof.</div>
      </aside>

      <main className="content">
        <header className="topbar">
          <div><span className="eyebrow">AI FORENSICS CONSOLE</span><h1>Detection workspace</h1><p>Inspect an image and review a clear, explainable model assessment.</p></div>
          <div className={`api-pill ${apiOnline ? 'online' : ''}`}><span /> {apiOnline ? 'API connected' : 'API offline'}<small>{API.replace('http://', '')}</small></div>
        </header>

        <section className="hero-grid">
          <div className="card upload-card">
            <div className="section-head"><div><span className="label">IMAGE ANALYSIS</span><h2>Upload media</h2><p>Drop an image here or choose a file from your device.</p></div><span className="step">01</span></div>
            <label className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); selectFile(e.dataTransfer.files[0]); }}>
              {preview ? <img className="preview" src={preview} alt="Selected media preview" /> : <div className="drop-content"><div className="upload-icon">↑</div><strong>Drag & drop an image</strong><span>or click to browse</span><small>JPG, PNG, WEBP</small></div>}
              <input type="file" accept="image/*" onChange={e => selectFile(e.target.files?.[0])} />
            </label>
            {file && <div className="file-row"><span>{file.name}</span><button onClick={reset}>Remove</button></div>}
            <button className="primary" disabled={!file || busy || !apiOnline} onClick={analyze}>{busy ? 'Analyzing…' : 'Run detection'}</button>
          </div>

          <div className={`card result-card ${result && !result.error ? verdictClass : ''}`}>
            <div className="section-head"><div><span className="label">ASSESSMENT</span><h2>Forensic result</h2><p>Values below come directly from the DEEP-Guard API.</p></div><span className="step">02</span></div>
            {!result ? <div className="result-empty"><div className="scan-icon">◈</div><strong>Ready for inspection</strong><span>Upload an image to generate a model assessment.</span></div> : result.error ? <div className="error-box">{result.error}</div> : <div className="result-body">
              <div className="verdict-row"><div><span className="result-label">VERDICT</span><div className="verdict">{label}</div></div><div className="confidence"><strong>{confidence}%</strong><span>model score</span></div></div>
              <div className="meter"><span style={{ width: `${confidence ?? 0}%` }} /></div>
              <div className="details"><div><span>Status</span><strong>{result.status || '—'}</strong></div><div><span>Model</span><strong>{result.model || '—'}</strong></div><div><span>Modality</span><strong>{result.modality || '—'}</strong></div></div>
              <div className="evidence"><span className="result-label">EVIDENCE RETURNED BY API</span><ul>{(result.evidence || []).map((x, i) => <li key={i}>{x}</li>)}</ul></div>
            </div>}
          </div>
        </section>

        {explanation && <section className="card explanation-card">
          <div className="section-head explanation-head"><div><span className="label">EXPLAINED SUMMARY</span><h2>What this result means</h2><p>A plain-language interpretation of the current model output.</p></div><span className="explain-badge">{isReal ? 'REAL SIGNAL' : 'AI SIGNAL'}</span></div>
          <div className="explanation-grid">
            <article className="explanation-main"><h3>{explanation.title}</h3><p>{explanation.summary}</p></article>
            <article className="explain-block"><span>01 · CONFIDENCE</span><p>{explanation.confidence}</p></article>
            <article className="explain-block"><span>02 · WHAT WAS RETURNED</span><ul>{explanation.evidence.map((x, i) => <li key={i}>{x}</li>)}</ul></article>
            <article className="explain-block"><span>03 · CURRENT LIMITATION</span><p>{explanation.limitation}</p></article>
            <article className="explain-block"><span>04 · RECOMMENDED NEXT STEP</span><p>{explanation.next}</p></article>
          </div>
        </section>}

        <section className="stats">
          <div className="stat card"><span>Analyses</span><strong>{history.length}</strong><small>stored locally</small></div>
          <div className="stat card"><span>Model</span><strong>EfficientNet-B0</strong><small>image detector</small></div>
          <div className="stat card"><span>Pipeline</span><strong>Ready</strong><small>{apiOnline ? 'API responding' : 'Start FastAPI'}</small></div>
        </section>

        <section className="card history-card">
          <div className="section-head"><div><span className="label">HISTORY</span><h2>Recent analyses</h2></div><button className="ghost" onClick={() => { setHistory([]); localStorage.removeItem(HISTORY_KEY); }}>Clear</button></div>
          {history.length === 0 ? <p className="muted">Your recent results will appear here after an analysis.</p> : <div className="history-list">{history.map(item => <div className="history-item" key={item.id}><div className={`dot ${item.verdict === 'likely_real' ? 'real' : 'ai'}`} /><div><strong>{item.verdict === 'likely_real' ? 'Likely Real' : 'AI Generated'}</strong><span>{item.filename || 'Image'} · {item.time}</span></div><b>{item.confidence == null ? '—' : `${(item.confidence * 100).toFixed(1)}%`}</b></div>)}</div>}
        </section>

        <footer>DEEP-Guard · Research prototype · Predictions are assessments, not definitive proof.</footer>
      </main>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
