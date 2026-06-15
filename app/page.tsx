'use client';

import { useState } from 'react';

export default function Dashboard() {
  const [trainingFiles, setTrainingFiles] = useState<FileList | null>(null);
  const [trainStatus, setTrainStatus] = useState<string>('');
  const [testImage, setTestImage] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTrain = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!trainingFiles) return;

    setTrainStatus('Extracting custom reference profiles...');
    const formData = new FormData();
    formData.append('mode', 'train');
    for (let i = 0; i < trainingFiles.length; i++) {
      formData.append('files', trainingFiles[i]);
    }

    try {
      const res = await fetch('/api/predict', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.success) {
        setTrainStatus(`✅ Registered! Memory bank built with ${data.registered_patches} reference patches.`);
      } else {
        setTrainStatus(`❌ Failed: ${data.error}`);
      }
    } catch (err: any) {
      setTrainStatus(`❌ Error compiling features.`);
    }
  };

  const handleTestUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setTestImage(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append('mode', 'test');
    formData.append('file', file);

    try {
      const response = await fetch('/api/predict', { method: 'POST', body: formData });
      const data = await response.json();
      if (response.ok) {
        setResult(data);
      } else {
        throw new Error(data.error || 'Inference processing error.');
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-slate-900 text-slate-100 min-h-screen font-sans">
      <header className="border-b border-slate-800 bg-slate-950/50 p-5 flex justify-between items-center">
        <h1 className="text-xl font-bold tracking-wider text-cyan-400 font-mono">🏭 ANY-OBJECT ANY-TIME VISION APP</h1>
        <span className="bg-cyan-500/10 text-cyan-400 text-xs px-3 py-1 rounded-full border border-cyan-500/20 font-mono">Status: Dynamic Target Memory</span>
      </header>

      <main className="max-w-7xl mx-auto p-6 md:p-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Multi-Object Dynamic Setup */}
        <div className="lg:col-span-1 space-y-6">
          
          {/* Step 1: Enroll Target Object */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-6 space-y-4 shadow-xl">
            <h2 className="text-md font-bold text-slate-200">Step 1: Enroll Your Reference Object</h2>
            <p className="text-slate-400 text-xs">Upload 2 to 5 images of *any* item in its perfect condition to generate an on-the-fly coreset representation bank.</p>
            
            <form onSubmit={handleTrain} className="space-y-3">
              <input type="file" multiple onChange={(e) => setTrainingFiles(e.target.files)} className="block w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-xs file:font-semibold file:bg-cyan-500/10 file:text-cyan-400 hover:file:bg-cyan-500/20" />
              <button type="submit" className="w-full bg-cyan-600 hover:bg-cyan-500 text-slate-950 text-xs font-bold py-2 rounded-lg transition">Compile Custom Memory Bank</button>
            </form>
            {trainStatus && <p className="text-xs font-mono text-cyan-400 bg-slate-900/50 p-3 rounded border border-slate-800">{trainStatus}</p>}
          </div>

          {/* Step 2: Test Frame Upload */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-6 space-y-4 shadow-xl">
            <h2 className="text-md font-bold text-slate-200">Step 2: Inspect Target Instance</h2>
            <label className="block border-2 border-dashed border-slate-700 hover:border-cyan-500 rounded-xl p-6 text-center cursor-pointer transition bg-slate-900/30">
              <input type="file" onChange={handleTestUpload} className="hidden" accept="image/*" />
              <div className="space-y-1">
                <span className="text-2xl block">🔍</span>
                <p className="text-xs font-medium text-slate-300">Drop your evaluation image here</p>
              </div>
            </label>

            {error && <div className="bg-red-500/10 text-red-400 border border-red-500/20 text-xs p-3 rounded font-mono">⚠️ {error}</div>}

            {result && (
              <div className="border border-slate-800 bg-slate-900/40 rounded-xl p-4 space-y-3">
                <div className="flex justify-between items-center py-1 border-b border-slate-800/60">
                  <span className="text-slate-400 text-xs">Verdict:</span>
                  <span className={`px-2 py-0.5 rounded text-xs font-bold border ${result.defect_detected ? 'bg-red-500/10 text-red-400 border-red-500/20' : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'}`}>{result.defect_detected ? 'ANOMALY DETECTED' : 'NORMAL COMPLIANT'}</span>
                </div>
                <div className="flex justify-between items-center py-1">
                  <span className="text-slate-400 text-xs">Distance Discrepancy:</span>
                  <span className="font-mono font-bold text-sm text-slate-200">{result.anomaly_score.toFixed(4)}</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Columns: Inspection Viewports */}
        <div className="lg:col-span-2">
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-6 min-h-[450px] flex flex-col justify-between shadow-2xl">
            <h2 className="text-md font-semibold text-slate-200 mb-4">Live Quality Control Monitor</h2>
            {loading ? (
              <div className="my-auto text-center py-20 text-cyan-400 font-mono animate-pulse tracking-wider text-xs">⚙️ COMPARING INPUT PATCHES AGAINST CUSTOM MEMORY BANK...</div>
            ) : testImage ? (
              <div className="max-w-md mx-auto space-y-2 w-full">
                <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Evaluation Sample Frame</span>
                <div className="border border-slate-800 rounded-lg overflow-hidden bg-slate-900 shadow-md">
                  <img src={testImage} alt="Evaluation specimen" className="w-full h-auto object-cover" />
                </div>
              </div>
            ) : (
              <div className="text-center py-24 text-slate-600 my-auto">
                <span className="text-4xl block mb-2 opacity-40">📸</span>
                <p className="text-slate-400 text-xs">Enroll a target type and supply a test frame to begin evaluation.</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}