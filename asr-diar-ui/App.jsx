import React, { useEffect, useMemo, useRef, useState } from "react";

// ===== ASR + Diarization Frontend =====
// Assumes a backend (e.g., FastAPI) that exposes:
//   POST   /process            { input_dir: string } -> { job_id: string }
//   GET    /status/:job_id     -> { state: "queued"|"running"|"done"|"error", progress?: number, log?: string[], output_txt?: string, error?: string }
//   GET    /file?path=...      -> serves the generated TXT (optional convenience)
// You can change BACKEND_URL below to match your server address.

const BACKEND_URL = "http://localhost:8000"; // <- chỉnh nếu backend chạy port khác

export default function App() {
  const [inputDir, setInputDir] = useState("D:/Test/data_cuochop/Hop01/input");
  const [jobId, setJobId] = useState(null);
  const [state, setState] = useState("idle");
  const [progress, setProgress] = useState<number>(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [outputTxt, setOutputTxt] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [polling, setPolling] = useState(false);
  const intervalRef = useRef(null);

  // Auto-scroll log view
  const logEndRef = useRef(null);
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const canRun = useMemo(() => {
    return inputDir.trim().length > 0 && (state === "idle" || state === "done" || state === "error");
  }, [inputDir, state]);

  const clearPolling = () => {
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setPolling(false);
  };

  const startPolling = (jid) => {
    setPolling(true);
    intervalRef.current = window.setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/status/${encodeURIComponent(jid)}`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        // Expected shape: { state, progress?, log?, output_txt?, error? }
        if (Array.isArray(data.log)) setLogs(data.log);
        if (typeof data.progress === "number") setProgress(Math.max(0, Math.min(100, data.progress)));
        if (data.output_txt) setOutputTxt(data.output_txt);
        if (data.state === "done" || data.state === "error") {
          setState(data.state);
          if (data.error) setErrorMsg(String(data.error));
          clearPolling();
        } else {
          setState(data.state || "running");
        }
      } catch (err) {
        console.error(err);
        setState("error");
        setErrorMsg(err?.message || String(err));
        clearPolling();
      }
    }, 1500);
  };

  const onRun = async () => {
    if (!canRun) return;
    setErrorMsg(null);
    setOutputTxt(null);
    setLogs([]);
    setProgress(0);
    setState("queued");
    try {
      const res = await fetch(`${BACKEND_URL}/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input_dir: inputDir })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const jid = String(data.job_id || "");
      if (!jid) throw new Error("job_id rỗng từ backend");
      setJobId(jid);
      setState("running");
      startPolling(jid);
    } catch (err) {
      setState("error");
      setErrorMsg(err?.message || String(err));
      clearPolling();
    }
  };

  const onCancel = async () => {
    if (!jobId) return;
    try {
      await fetch(`${BACKEND_URL}/cancel/${encodeURIComponent(jobId)}`, { method: "POST" });
    } catch {}
    clearPolling();
    setState("idle");
  };

  return (
    <div className="min-h-screen w-full bg-neutral-50 text-neutral-900 p-6">
      <div className="max-w-5xl mx-auto space-y-6">
        <header className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">ASR + Diarization Dashboard</h1>
          <a
            className="text-sm underline opacity-70 hover:opacity-100"
            href="#" onClick={(e)=>{e.preventDefault(); alert("Nhập đường dẫn thư mục input chứa \n- file audio: *.mp3 / *.wav / ... \n- file label.txt (tùy chọn).\nKết quả TXT sẽ lưu ở thư mục output cùng cấp.");}}
          >Hướng dẫn</a>
        </header>

        <section className="grid gap-4 md:grid-cols-5">
          <div className="md:col-span-4">
            <label className="text-sm font-medium">Input folder</label>
            <input
              className="mt-1 w-full rounded-xl border border-neutral-300 px-3 py-2 focus:outline-none focus:ring-4 focus:ring-neutral-200"
              placeholder="D:/Test/data_cuochop/Hop01/input"
              value={inputDir}
              onChange={(e)=>setInputDir(e.target.value)}
            />
            <p className="text-xs text-neutral-500 mt-1">Thư mục chứa audio họp và file label.txt (tùy chọn).</p>
          </div>
          <div className="md:col-span-1 flex items-end">
            <button
              onClick={onRun}
              disabled={!canRun}
              className={`w-full rounded-xl px-4 py-2 font-semibold shadow ${canRun?"bg-black text-white hover:opacity-90":"bg-neutral-300 text-neutral-600 cursor-not-allowed"}`}
            >
              {state === "running" || state === "queued" ? "Đang chạy" : "Chạy"}
            </button>
          </div>
        </section>

        <section className="grid gap-4 md:grid-cols-3">
          <div className="md:col-span-2 bg-white rounded-2xl shadow p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="font-semibold">Trạng thái</h2>
              <div className="text-sm">
                <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1 border ${stateBadgeClass(state)}`}>
                  <span className={`h-2 w-2 rounded-full ${dotClass(state)}`}></span>
                  {state.toUpperCase()}
                </span>
              </div>
            </div>

            <div className="h-2 w-full bg-neutral-200 rounded-full overflow-hidden">
              <div className="h-full bg-neutral-900 transition-all" style={{ width: `${progress}%` }} />
            </div>

            {outputTxt && (
              <div className="mt-4 text-sm">
                <div className="font-medium">Kết quả:</div>
                <div className="mt-1 break-all">
                  <a className="underline" href={`${BACKEND_URL}/file?path=${encodeURIComponent(outputTxt)}`} target="_blank" rel="noreferrer">
                    {outputTxt}
                  </a>
                </div>
              </div>
            )}

            {errorMsg && (
              <div className="mt-4 text-sm text-red-600">Lỗi: {errorMsg}</div>
            )}

            {(state === "running" || state === "queued") && (
              <div className="mt-4 flex gap-2">
                <button onClick={onCancel} className="rounded-xl border px-3 py-2 hover:bg-neutral-50">Huỷ</button>
              </div>
            )}
          </div>

          <div className="md:col-span-1 bg-white rounded-2xl shadow p-4">
            <h2 className="font-semibold mb-2">Log</h2>
            <div className="h-80 overflow-auto text-xs font-mono whitespace-pre-wrap border rounded-xl p-2 bg-neutral-50">
              {(logs?.length ? logs : ["(Chưa có log)"]).map((line, i) => (
                <div key={i}>{line}</div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>
        </section>

        <footer className="text-xs text-neutral-500 pt-4">
          Backend: {BACKEND_URL}
        </footer>
      </div>
    </div>
  );
}

function stateBadgeClass(state) {
  switch (state) {
    case "queued": return "border-yellow-300 text-yellow-700 bg-yellow-50";
    case "running": return "border-blue-300 text-blue-700 bg-blue-50";
    case "done": return "border-green-300 text-green-700 bg-green-50";
    case "error": return "border-red-300 text-red-700 bg-red-50";
    default: return "border-neutral-300 text-neutral-700 bg-neutral-50";
  }
}

function dotClass(state) {
  switch (state) {
    case "queued": return "bg-yellow-500";
    case "running": return "bg-blue-600";
    case "done": return "bg-green-600";
    case "error": return "bg-red-600";
    default: return "bg-neutral-400";
  }
}
