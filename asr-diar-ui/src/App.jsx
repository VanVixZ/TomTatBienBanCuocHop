import React, { useEffect, useMemo, useRef, useState } from "react";

const BACKEND_URL = "http://localhost:8000"; // đổi nếu backend khác cổng

export default function App() {
  const [inputDir, setInputDir] = useState("D:/Test/data_cuochop/Hop01/input");
  const [jobId, setJobId] = useState(null);
  const [state, setState] = useState("idle"); // idle | queued | running | done | error
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
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
    <div className="min-h-screen w-full" style={{ background: "#fafafa", color: "#111", padding: 24 }}>
      <div style={{ maxWidth: 1000, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h1 style={{ fontSize: 22, fontWeight: 700 }}>ASR + Diarization Dashboard</h1>
          <a
            href="#"
            onClick={(e) => { e.preventDefault(); alert("Nhập đường dẫn thư mục input chứa:\n- file audio (*.mp3 / *.wav / ...)\n- file label.txt (tùy chọn)\nKết quả TXT sẽ lưu ở thư mục output cùng cấp."); }}
            style={{ fontSize: 13, textDecoration: "underline", opacity: 0.8 }}
          >
            Hướng dẫn
          </a>
        </header>

        <section style={{ display: "grid", gap: 12, gridTemplateColumns: "1fr auto", marginBottom: 16 }}>
          <div>
            <label style={{ fontSize: 13, fontWeight: 600 }}>Input folder</label>
            <input
              style={{ display: "block", marginTop: 6, width: "100%", borderRadius: 12, border: "1px solid #d4d4d4", padding: "10px 12px" }}
              placeholder="D:/Test/data_cuochop/Hop01/input"
              value={inputDir}
              onChange={(e) => setInputDir(e.target.value)}
            />
            <div style={{ fontSize: 12, color: "#666", marginTop: 6 }}>Thư mục chứa audio họp và file label.txt (tùy chọn).</div>
          </div>
          <div style={{ display: "flex", alignItems: "end" }}>
            <button
              onClick={onRun}
              disabled={!canRun}
              style={{
                borderRadius: 12,
                padding: "10px 16px",
                fontWeight: 600,
                boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
                background: canRun ? "#111" : "#d4d4d4",
                color: canRun ? "#fff" : "#666",
                cursor: canRun ? "pointer" : "not-allowed"
              }}
            >
              {state === "running" || state === "queued" ? "Đang chạy" : "Chạy"}
            </button>
          </div>
        </section>

        <section style={{ display: "grid", gap: 12, gridTemplateColumns: "2fr 1fr" }}>
          <div style={{ background: "#fff", borderRadius: 16, boxShadow: "0 1px 6px rgba(0,0,0,0.06)", padding: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 12 }}>
              <div style={{ fontWeight: 600 }}>Trạng thái</div>
              <span style={{
                border: "1px solid #ddd",
                borderRadius: 999,
                padding: "4px 10px",
                fontSize: 13,
                background: badgeBg(state),
                color: badgeText(state),
                borderColor: badgeBorder(state)
              }}>
                ● {state.toUpperCase()}
              </span>
            </div>

            <div style={{ height: 8, width: "100%", background: "#e5e5e5", borderRadius: 999, overflow: "hidden" }}>
              <div style={{ height: "100%", background: "#111", width: `${progress}%`, transition: "width .3s" }} />
            </div>

            {outputTxt && (
              <div style={{ marginTop: 12, fontSize: 14 }}>
                <div style={{ fontWeight: 600 }}>Kết quả:</div>
                <div style={{ marginTop: 4, wordBreak: "break-all" }}>
                  <a href={`${BACKEND_URL}/file?path=${encodeURIComponent(outputTxt)}`} target="_blank" rel="noreferrer">
                    {outputTxt}
                  </a>
                </div>
              </div>
            )}

            {errorMsg && (
              <div style={{ marginTop: 12, fontSize: 14, color: "#d11" }}>Lỗi: {errorMsg}</div>
            )}

            {(state === "running" || state === "queued") && (
              <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
                <button onClick={onCancel} style={{ borderRadius: 12, border: "1px solid #ddd", padding: "8px 12px" }}>
                  Huỷ
                </button>
              </div>
            )}
          </div>

          <div style={{ background: "#fff", borderRadius: 16, boxShadow: "0 1px 6px rgba(0,0,0,0.06)", padding: 16 }}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>Log</div>
            <div style={{ height: 320, overflow: "auto", fontSize: 12, fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace", whiteSpace: "pre-wrap", border: "1px solid #eee", borderRadius: 12, padding: 8, background: "#fafafa" }}>
              {(logs && logs.length ? logs : ["(Chưa có log)"]).map((line, i) => (
                <div key={i}>{line}</div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>
        </section>

        <footer style={{ fontSize: 12, color: "#666", paddingTop: 16 }}>
          Backend: {BACKEND_URL}
        </footer>
      </div>
    </div>
  );
}

function badgeBg(s) {
  if (s === "queued") return "#FFFBEB";
  if (s === "running") return "#EFF6FF";
  if (s === "done") return "#ECFDF5";
  if (s === "error") return "#FEF2F2";
  return "#F5F5F5";
}
function badgeText(s) {
  if (s === "queued") return "#92400E";
  if (s === "running") return "#1D4ED8";
  if (s === "done") return "#047857";
  if (s === "error") return "#B91C1C";
  return "#374151";
}
function badgeBorder(s) {
  if (s === "queued") return "#FCD34D";
  if (s === "running") return "#93C5FD";
  if (s === "done") return "#86EFAC";
  if (s === "error") return "#FCA5A5";
  return "#D1D5DB";
}
