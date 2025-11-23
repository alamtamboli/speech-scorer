// frontend/vite-project/src/components/ScorePanel.jsx
import React, { useMemo, useState } from "react";
import RadarChart from "./RadarChart";

function ProgressBar({ value }) {
  const pct = Math.round(value);
  return (
    <div style={{ background: "#eee", borderRadius: 6, overflow: "hidden", height: 14 }}>
      <div style={{
        width: `${pct}%`,
        height: "100%",
        background: pct >= 75 ? "#16a34a" : pct >= 50 ? "#f59e0b" : "#ef4444",
        transition: "width 400ms ease"
      }} />
    </div>
  );
}

export default function ScorePanel({ quick, result }) {
  const [showDebug, setShowDebug] = useState(false);

  const perCriteria = result?.per_criteria || [];

  // build radar labels & values
  const { labels, values } = useMemo(() => {
    const ordered = perCriteria.slice();
    // keep ordering sensible: content, speech_rate, language_grammar, clarity, engagement
    // We'll just map them in the returned order for now.
    const labs = ordered.map(c => c.name);
    const vals = ordered.map(c => Math.round((c.score || 0) * 100));
    return { labels: labs, values: vals };
  }, [perCriteria]);

  const overall = result?.overall ?? null;

  return (
    <div>
      <div className="score-card" style={{ marginBottom: 16 }}>
        <h3 style={{ margin: 0 }}>Results</h3>
        <div style={{ fontSize: 28, fontWeight: 700, marginTop: 8 }}>
          Overall Score: {overall !== null ? `${overall}%` : "—"}
        </div>
      </div>

      <div className="score-card">
        <h4 style={{ marginTop: 0 }}>Visual Summary</h4>
        <div style={{ display: "flex", gap: 20 }}>
          <div style={{ width: 420 }}>
            <RadarChart labels={labels} data={values} />
          </div>

          <div style={{ flex: 1 }}>
            {perCriteria.length === 0 ? (
              <div>No results yet. Enter transcript and click Score.</div>
            ) : (
              perCriteria.map((c) => (
                <div key={c.id} style={{ marginBottom: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <div style={{ fontWeight: 600 }}>{c.name}</div>
                    <div style={{ fontSize: 13, color: "#334155" }}>{Math.round(c.score * 100)}% {c.band ? <span style={{ marginLeft: 8, fontSize: 12, background: "#eef2ff", padding: "2px 8px", borderRadius: 12 }}>{c.band}</span> : null}</div>
                  </div>
                  <ProgressBar value={(c.score || 0) * 100} />
                  {/* optional small details */}
                  <div style={{ fontSize: 12, color: "#64748b", marginTop: 6 }}>
                    <button style={{ border: "none", background: "transparent", color: "#2563eb", cursor: "pointer" }} onClick={(e) => {
                      const el = e.target.parentElement.nextSibling;
                      if (el) el.style.display = el.style.display === "block" ? "none" : "block";
                    }}>▸ More details</button>
                  </div>
                  <div style={{ display: "none", marginTop: 8, fontSize: 13, color: "#334155" }}>
                    <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{JSON.stringify(c.components || {}, null, 2)}</pre>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="score-card">
        <h4 style={{ marginTop: 0 }}>Detailed Breakdown</h4>
        {perCriteria.length === 0 && <div>No details available.</div>}
        {perCriteria.map(c => (
          <div key={c.id} style={{ padding: "8px 0", borderBottom: "1px solid #f1f5f9" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div><strong>{c.name}</strong></div>
              <div style={{ color: "#0f172a", fontWeight: 700 }}>{Math.round(c.score * 100)}%</div>
            </div>
            <div style={{ marginTop: 6, color: "#475569", fontSize: 13 }}>
              <div>{c.components && Object.keys(c.components).length > 0 ? (
                <div>
                  <em>Components:</em>
                  <ul style={{ marginTop: 6 }}>
                    {Object.entries(c.components).map(([k, v]) => <li key={k}><strong>{k}:</strong> {String(v)}</li>)}
                  </ul>
                </div>
              ) : <span>No component data.</span>}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="score-card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h4 style={{ margin: 0 }}>Debug / Raw Metrics</h4>
          <button onClick={() => setShowDebug(prev => !prev)} style={{ background: "#eef2ff", border: "none", padding: "6px 10px", borderRadius: 8, cursor: "pointer" }}>{showDebug ? "Hide" : "Show"}</button>
        </div>

        {showDebug && (
          <div style={{ marginTop: 12, fontSize: 13, color: "#0f172a" }}>
            <pre style={{ whiteSpace: "pre-wrap" }}>
{JSON.stringify({
  overall,
  perCriteria,
  quick
}, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
