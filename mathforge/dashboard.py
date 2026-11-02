"""Live dashboard: a small local web page that follows a run while it is going.

    mathforge dashboard --run-dir runs/laptop          # then open http://127.0.0.1:8765

Read-only and dependency-free (standard-library HTTP server, inline JavaScript and
SVG). It reads the run's SQLite database on every refresh; WAL mode lets it do that
safely while the orchestrator keeps writing.
"""

from __future__ import annotations

import json
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .report import CURVES


def snapshot(db_path: Path, recent: int = 12) -> dict:
    """Everything the page shows, as JSON-ready data."""
    db = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=10)
    try:
        engines: dict[str, dict] = {}
        for e, name, r, v in db.execute("SELECT engine, name, round, value FROM metrics ORDER BY round"):
            engines.setdefault(e, {"curves": {}})["curves"].setdefault(name, []).append([r, v])
        for e, n in db.execute("SELECT engine, COUNT(*) FROM discoveries GROUP BY engine"):
            engines.setdefault(e, {"curves": {}})["count"] = n
        rows = db.execute("SELECT id, engine, kind, statement, payload, round FROM discoveries "
                          "ORDER BY id DESC LIMIT ?", (recent,)).fetchall()
        last = db.execute("SELECT MAX(round), MAX(t) FROM metrics").fetchone()
    finally:
        db.close()
    out_recent = []
    for i, e, kind, st, payload, r in rows:
        p = json.loads(payload)
        out_recent.append({"id": i, "engine": e, "kind": kind, "statement": st, "round": r,
                           "verified": p.get("verified") or (f"{p['digits']} digits" if "digits" in p else kind)})
    return {"round": last[0] or 0, "updated": last[1], "engines": engines, "recent": out_recent,
            "featured": [[e, m, t, cum] for e, m, t, cum in CURVES]}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MathForge live</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#3b6ea8;--grid:#e4e4df}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecece8;--mute:#9a9a93;--line:#7aa7dd;--grid:#33332f}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
header{padding:16px 20px;display:flex;gap:24px;align-items:baseline;flex-wrap:wrap}
h1{font-size:18px;margin:0}.stat{color:var(--mute)}.stat b{color:var(--ink);font-size:16px}
main{padding:0 20px 24px;display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}
.card{background:var(--card);border-radius:8px;padding:12px 14px;box-shadow:0 1px 2px rgba(0,0,0,.06)}
.card h2{font-size:13px;font-weight:600;margin:0 0 6px}svg{width:100%;height:150px;display:block}
.wide{grid-column:1/-1}table{width:100%;border-collapse:collapse;font-size:13px}
td{padding:5px 6px;border-top:1px solid var(--grid);vertical-align:top}td.m{color:var(--mute);white-space:nowrap}
code{font-size:12px;word-break:break-word}select{background:var(--card);color:var(--ink);border:1px solid var(--grid);border-radius:4px}
</style></head><body>
<header><h1>MathForge</h1><span class="stat">round <b id="round">-</b></span>
<span class="stat">verified results <b id="total">-</b></span><span class="stat" id="per"></span>
<span class="stat">add chart <select id="pick"><option value="">metric...</option></select></span>
<span class="stat" id="age"></span></header>
<main id="main"></main>
<script>
const extra = new Set();
function chart(title, pts, cum){
  if(cum){let a=0;pts=pts.map(([r,v])=>[r,a+=v]);title+=" (cumulative)";}
  const W=320,H=150,P=28,xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(0,...ys),y1=Math.max(...ys);
  const sx=x=>P+(W-P-6)*(x1>x0?(x-x0)/(x1-x0):.5), sy=y=>H-18-(H-26)*(y1>y0?(y-y0)/(y1-y0):.5);
  const d=pts.map((p,i)=>(i?"L":"M")+sx(p[0]).toFixed(1)+" "+sy(p[1]).toFixed(1)).join(" ");
  const last=ys[ys.length-1], fmt=v=>Math.abs(v)>=100||Number.isInteger(v)?Math.round(v):v.toFixed(3);
  return `<div class="card"><h2>${title} <span class="stat">${fmt(last)}</span></h2><svg viewBox="0 0 ${W} ${H}">
  <line x1="${P}" y1="${H-18}" x2="${W-6}" y2="${H-18}" stroke="var(--grid)"/>
  <text x="${P-4}" y="${sy(y1)+4}" font-size="10" text-anchor="end" fill="var(--mute)">${fmt(y1)}</text>
  <text x="${P-4}" y="${H-14}" font-size="10" text-anchor="end" fill="var(--mute)">${fmt(y0)}</text>
  <text x="${W-6}" y="${H-3}" font-size="10" text-anchor="end" fill="var(--mute)">round ${x1}</text>
  <path d="${d}" fill="none" stroke="var(--line)" stroke-width="1.8"/></svg></div>`;
}
const esc=s=>String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
async function refresh(){
  let s; try{ s=await (await fetch("api/state")).json(); }catch(e){ document.getElementById("age").textContent="(no connection)"; return; }
  document.getElementById("round").textContent=s.round;
  let total=0, per=[]; for(const [e,v] of Object.entries(s.engines)){ total+=v.count||0; per.push(`${e} ${v.count||0}`); }
  document.getElementById("total").textContent=total; document.getElementById("per").textContent=per.join(" · ");
  if(s.updated) document.getElementById("age").textContent=`updated ${Math.round(Date.now()/1000-s.updated)} s ago`;
  const pick=document.getElementById("pick"), have=new Set([...pick.options].map(o=>o.value));
  for(const [e,v] of Object.entries(s.engines)) for(const m of Object.keys(v.curves)) if(!have.has(e+"/"+m)){
    const o=document.createElement("option"); o.value=o.textContent=e+"/"+m; pick.appendChild(o); }
  let html="";
  for(const [e,m,t,cum] of s.featured){ const c=s.engines[e]&&s.engines[e].curves[m]; if(c&&c.length>1) html+=chart(t,c,cum); }
  for(const k of extra){ const [e,m]=k.split("/"); const c=s.engines[e]&&s.engines[e].curves[m]; if(c&&c.length) html+=chart(k,c,false); }
  html+=`<div class="card wide"><h2>Latest verified results</h2><table>`+s.recent.map(d=>
    `<tr><td class="m">#${d.id} · r${d.round}<br>${esc(d.engine)}</td><td><code>${esc(d.statement)}</code><br><span class="stat">${esc(d.verified)}</span></td></tr>`).join("")+`</table></div>`;
  document.getElementById("main").innerHTML=html;
}
document.getElementById("pick").onchange=e=>{ if(e.target.value){ extra.add(e.target.value); e.target.value=""; refresh(); } };
refresh(); setInterval(refresh, 5000);
</script></body></html>"""


def make_handler(db_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            if self.path.split("?")[0] in ("/", "/index.html"):
                body, ctype = PAGE.encode("utf-8"), "text/html; charset=utf-8"
            elif self.path.startswith("/api/state"):
                try:
                    body, ctype = json.dumps(snapshot(db_path)).encode("utf-8"), "application/json"
                except sqlite3.Error as e:
                    self.send_error(503, f"database not readable yet: {e}")
                    return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the terminal quiet
            pass

    return Handler


def serve(run_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    db = run_dir / "mathforge.db"
    if not db.exists():
        raise SystemExit(f"No run found at {run_dir}. Start one with: mathforge run")
    server = ThreadingHTTPServer((host, port), make_handler(db))
    print(f"MathForge dashboard for {run_dir}: http://{host}:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
