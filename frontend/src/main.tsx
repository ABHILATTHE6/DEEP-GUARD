import React, {useState} from 'react';
import {createRoot} from 'react-dom/client';
import './styles.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App(){
  const [file,setFile]=useState<File|null>(null); const [result,setResult]=useState<any>(null); const [busy,setBusy]=useState(false);
  async function analyze(){ if(!file)return; setBusy(true); setResult(null); const fd=new FormData(); fd.append('media',file); try{ const r=await fetch(`${API}/api/v1/analyze`,{method:'POST',body:fd}); const data=await r.json(); setResult(r.ok?data:{error:data.detail||'Analysis failed'}); }catch(e){setResult({error:'API unavailable. Start FastAPI on port 8000.'});}finally{setBusy(false)} }
  return <main><header><div><span className="eyebrow">MEDIA AUTHENTICITY LAB</span><h1>DEEP-Guard</h1><p>Multimodal AI-generated media and deepfake assessment.</p></div><span className="pill">Research Prototype</span></header>
  <section className="grid"><div className="card upload"><h2>Analyze media</h2><p>Upload an image, video, or audio file.</p><input type="file" accept="image/*,video/*,audio/*" onChange={e=>{setFile(e.target.files?.[0]||null);setResult(null)}}/><button disabled={!file||busy} onClick={analyze}>{busy?'Analyzing…':'Analyze with DEEP-Guard'}</button>{file&&<div className="file">{file.name} · {(file.size/1024/1024).toFixed(2)} MB</div>}</div>
  <div className="card result"><h2>Assessment</h2>{!result?<div className="empty">Upload media to begin.</div>:result.error?<div className="warning">{result.error}</div>:<><div className="verdict">{String(result.verdict||'uncertain').replaceAll('_',' ').toUpperCase()}</div><div className="meta"><b>Confidence</b><span>{result.confidence==null?'—':`${(result.confidence*100).toFixed(1)}%`}</span></div><div className="meta"><b>Modality</b><span>{result.modality}</span></div><div className="meta"><b>Model</b><span>{result.model}</span></div><div className="status">{result.status}</div><h3>Evidence</h3><ul>{(result.evidence||[]).map((x:string,i:number)=><li key={i}>{x}</li>)}</ul></>}</div></section>
  <section className="card philosophy"><h2>DEEP-Guard evidence model</h2><div className="features"><span>Deep Learning</span><span>Explainability</span><span>Robustness</span><span>Multimodal Fusion</span><span>Uncertainty</span></div><p>Predictions are assessments, not proof. Untrained modalities are explicitly reported as unavailable rather than returning fabricated scores.</p></section></main>
}
createRoot(document.getElementById('root')!).render(<App/>);
