# SPDX-License-Identifier: Apache-2.0
"""Self-contained HTML for the local Malibu Pearl control page."""
from __future__ import annotations

import json


_CSS = r"""
@import url("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600;700&family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap");
:root{
  --bg:#f6f7f8; --surface:#ffffff; --ink:#14181c; --muted:#5b6670; --line:#dfe3e7;
  --accent:#2f6f8f; --accent-ink:#ffffff; --nacre:linear-gradient(120deg,#e8eef3 0%,#f3ece6 35%,#e6f0ec 65%,#ece8f3 100%);
  --good:#2e7d4f; --warn:#a46a12; --bad:#a63d2f; --code-bg:#11161b; --code-fg:#d8e2ea;
  --f-display:"Bricolage Grotesque",ui-sans-serif,system-ui,sans-serif;
  --f-body:"Instrument Sans",ui-sans-serif,system-ui,-apple-system,sans-serif;
  --f-mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0e1215; --surface:#161b20; --ink:#e9edf0; --muted:#9aa6b0; --line:#2a3138;
  --accent:#6fb3d2; --accent-ink:#0b1216; --nacre:linear-gradient(120deg,#1a2128 0%,#221e1c 35%,#18231f 65%,#1f1c26 100%);
  --good:#5cc489; --warn:#e0a54a; --bad:#e07363; --code-bg:#0a0d10; --code-fg:#d8e2ea; color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:16px;line-height:1.55;padding:20px 16px 64px}
.wrap{max-width:900px;margin:0 auto}
header.top{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:20px}
.brand{font:700 20px/1 var(--f-display);letter-spacing:-.01em}
.brand span{color:var(--muted);font-weight:500}
.machine{color:var(--muted);font-size:14px;text-align:right}
h1{font:700 clamp(32px,5vw,50px)/1.05 var(--f-display);letter-spacing:-.025em;margin:0 0 14px;text-wrap:balance}
h2{font:700 24px/1.15 var(--f-display);letter-spacing:-.015em;margin:0 0 10px;text-wrap:balance}
h3{font:600 17px/1.3 var(--f-body);margin:0 0 6px}
p{margin:0 0 12px;max-width:68ch}
.hero{background:var(--nacre);border:1px solid var(--line);border-radius:18px;padding:26px;margin-bottom:20px}
.hero-row{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap}
.pill{font:600 12px var(--f-body);padding:5px 11px;border-radius:999px;display:inline-flex;gap:7px;align-items:center;background:color-mix(in srgb,var(--muted) 12%,transparent);color:var(--muted)}
.pill::before{content:"";width:8px;height:8px;border-radius:50%;background:currentColor}
.pill.on{color:var(--good);background:color-mix(in srgb,var(--good) 14%,transparent)}
.pill.warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 14%,transparent)}
.pill.bad{color:var(--bad);background:color-mix(in srgb,var(--bad) 14%,transparent)}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0 0}
.stats div,.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px}
.stats .v{font:600 22px var(--f-mono);font-variant-numeric:tabular-nums;word-break:break-word}
.stats .v.nospeed{font:500 16px var(--f-body);color:var(--muted);padding-top:4px}
.stats .l{font-size:13px;color:var(--muted)}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}
.card{padding:22px;margin-bottom:20px}
.ctl{display:grid;gap:16px;margin-top:16px}
.ctl .line{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
.sub{font-size:13px;color:var(--muted)}
.btn{display:inline-flex;align-items:center;gap:8px;font:600 15px var(--f-body);padding:10px 16px;border-radius:12px;border:1px solid transparent;cursor:pointer;background:var(--accent);color:var(--accent-ink)}
.btn.ghost{background:transparent;color:var(--ink);border-color:var(--line)}
.btn.danger{background:transparent;color:var(--warn);border-color:var(--line)}
.btn[disabled]{opacity:.55;cursor:default}
.btn:focus-visible,.seg button:focus-visible,.switch:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:10px;overflow:hidden}
.seg button{font:500 14px var(--f-body);padding:8px 14px;background:transparent;color:var(--ink);border:0;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--bg)}
.switch{width:46px;height:28px;border-radius:999px;border:1px solid var(--line);background:var(--bg);position:relative;cursor:pointer;padding:0}
.switch::after{content:"";position:absolute;top:3px;left:3px;width:20px;height:20px;border-radius:50%;background:var(--muted);transition:left .15s}
.switch[aria-checked="true"]{background:var(--accent);border-color:var(--accent)}
.switch[aria-checked="true"]::after{left:21px;background:var(--accent-ink)}
.applying{font-size:13px;color:var(--accent);min-height:1em}
.errline{font-size:13px;color:var(--warn);margin-top:6px;word-break:break-word}
.bar{height:10px;border-radius:999px;background:var(--bg);border:1px solid var(--line);overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent);border-radius:999px;width:0%}
.earn-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}
.metric{border-top:1px solid var(--line);padding-top:10px}
.metric .v{font:600 19px var(--f-mono);font-variant-numeric:tabular-nums}
.metric .l{font-size:13px;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:14px}
td,th{text-align:left;padding:8px 0;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--muted);font-size:12px;letter-spacing:.04em;text-transform:uppercase}
td.num,th.num{text-align:right}
td.num{font-family:var(--f-mono);font-variant-numeric:tabular-nums}
.tx{font-family:var(--f-mono);font-size:13px;color:var(--muted);max-width:28ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tablewrap{overflow-x:auto}
.confirm{border:1px solid var(--warn);border-radius:12px;padding:14px;margin-top:12px}
.removed{display:none;background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:28px;margin-top:20px}
.removed-mode>*:not(#removed){display:none!important}.removed-mode #removed{display:block}
.foot{margin-top:34px;padding-top:24px;border-top:1px solid var(--line);font-size:14px}
.fcols{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}
.fcols>div{display:flex;flex-direction:column;gap:8px;min-width:0}
.flabel{font:500 11px var(--f-mono);letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:2px}
.foot a{color:var(--ink);text-decoration:none}
.foot a:hover,a:hover{color:var(--accent);text-decoration:underline}
a{color:var(--accent)}
[hidden]{display:none !important}
@media (max-width:720px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}.fcols{grid-template-columns:1fr 1fr}}
@media (max-width:480px){.stats,.earn-row{grid-template-columns:1fr}header.top{align-items:flex-start;flex-direction:column}.machine{text-align:left}}
"""


_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Malibu Pearl - This Mac</title>
<style>__PMK_CSS__</style>
</head>
<body>
<main class="wrap" id="app">
  <header class="top">
    <div class="brand">Malibu Pearl <span>&middot; This Mac</span></div>
    <div class="machine" id="computerName"></div>
  </header>

  <section class="hero">
    <div class="hero-row">
      <div>
        <h1 id="headline">This Mac is mining Pearl.</h1>
        <p class="sub">Pause, slow down or remove the miner here. Changes take effect right away. Bookmark this page, or go to localhost:47811 any time.</p>
      </div>
      <span class="pill" id="statusPill">Starting up</span>
    </div>
    <div class="stats">
      <div><div class="v" id="speed">Measuring&hellip;</div><div class="l">Speed</div></div>
      <div><div class="v" id="accepted">0</div><div class="l">Shares accepted</div></div>
      <div><div class="v" id="rejected">0</div><div class="l">Rejected</div></div>
      <div><div class="v" id="uptime">0s</div><div class="l">Uptime</div></div>
    </div>
  </section>

  <div class="grid">
    <section class="card">
      <h2>Controls</h2>
      <div class="ctl">
        <div class="line">
          <div><h3>Mining</h3><div class="sub" id="pauseSub">Pause or resume immediately.</div></div>
          <button class="btn" id="pauseButton" type="button">Pause</button>
        </div>
        <div class="line">
          <div><h3>How hard to work your Mac</h3><div class="sub">Low is quietest. Full is fastest.</div></div>
          <div class="seg" id="intensityGroup" role="group" aria-label="Intensity">
            <button type="button" data-intensity="low">Low</button>
            <button type="button" data-intensity="medium">Medium</button>
            <button type="button" data-intensity="full">Full</button>
          </div>
        </div>
        <div class="line">
          <div><h3>Only when plugged in</h3><div class="sub">Pause automatically on battery.</div></div>
          <button class="switch" id="onlyAc" type="button" role="switch" aria-checked="true" aria-label="Only when plugged in"></button>
        </div>
        <div class="line">
          <div><h3>Start when I log in</h3><div class="sub">Keeps mining after a restart.</div></div>
          <button class="switch" id="startAtLogin" type="button" role="switch" aria-checked="true" aria-label="Start when I log in"></button>
        </div>
        <div>
          <button class="btn danger" id="removeButton" type="button">Remove from this Mac</button>
          <div class="confirm" id="removeConfirm" hidden>
            <p>Remove Malibu Pearl from this Mac?</p>
            <button class="btn danger" id="confirmRemove" type="button">Yes, remove it</button>
            <button class="btn ghost" id="cancelRemove" type="button">Cancel</button>
          </div>
        </div>
        <div class="applying" id="applying"></div>
        <div class="errline" id="localError"></div>
      </div>
    </section>

    <section class="card">
      <h2>This Mac</h2>
      <div class="earn-row">
        <div class="metric"><div class="v" id="kernel">Standard</div><div class="l">Speed mode</div></div>
        <div class="metric"><div class="v" id="v4Ready">No</div><div class="l">Ready for Pearl's v4 upgrade</div></div>
        <div class="metric"><div class="v" id="power">-</div><div class="l">Power</div></div>
        <div class="metric"><div class="v" id="intensity">Full</div><div class="l">Intensity</div></div>
      </div>
    </section>
  </div>

  <section class="card">
    <div class="hero-row">
      <h2>Earnings</h2>
      <a id="heroLink" target="_blank" rel="noreferrer">Open on HeroMiners</a>
    </div>
    <p class="sub" id="earningsStatus">Loading earnings&hellip;</p>
    <div id="earningsBody">
      <div class="earn-row">
        <div class="metric"><div class="v" id="pendingBalance">0 PRL</div><div class="l">Waiting for payout (pays at 1 PRL)</div></div>
        <div class="metric"><div class="v" id="totalPaid">0 PRL</div><div class="l">Total paid</div></div>
        <div class="metric"><div class="v" id="poolHashrate">-</div><div class="l">Speed at the pool, whole wallet</div></div>
      </div>
      <div class="bar" aria-label="Progress to 1 PRL payout threshold"><i id="payoutProgress"></i></div>
      <div class="tablewrap">
        <table>
          <thead><tr><th>Payment</th><th class="num">Amount</th><th class="num">When</th></tr></thead>
          <tbody id="payments"></tbody>
        </table>
      </div>
    </div>
  </section>

  <section class="removed" id="removed">
    <h2>Removed.</h2>
    <p>You can close this page.</p>
  </section>

  <footer class="foot">
    <div class="fcols">
      <div><div class="flabel">Code</div><a href="https://github.com/Augustas11/pearl-apple-miner" rel="noreferrer">Miner on GitHub</a></div>
      <div><div class="flabel">Project</div><a href="https://malibu.tech/" rel="noreferrer">Malibu</a></div>
      <div><div class="flabel">Built by</div><a href="https://github.com/Augustas11" rel="noreferrer">Augustas11</a></div>
    </div>
  </footer>
</main>
<script>
const PMK = __PMK_CONFIG__;

const $ = (id) => document.getElementById(id);
const stateLabels = {
  checking: ["Starting up", ""],
  startup: ["Starting up", ""],
  mining: ["Mining", "on"],
  paused: ["Paused", "warn"],
  paused_battery: ["Paused on battery", "warn"],
  paused_thermal: ["Paused, Mac is hot", "warn"],
  error: ["Error", "bad"],
  uninstalled: ["Removed", "warn"]
};
let lastStatus = {};
let pending = null;
let removed = false;

function text(id, value) {
  $(id).textContent = value == null || value === "" ? "-" : String(value);
}

function prl(value) {
  const n = Number(value || 0) / 1e8;
  return n.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: n < 1 ? 4 : 2}) + " PRL";
}

function seconds(value) {
  let s = Math.max(0, Math.floor(Number(value || 0)));
  const d = Math.floor(s / 86400); s %= 86400;
  const h = Math.floor(s / 3600); s %= 3600;
  const m = Math.floor(s / 60); s %= 60;
  if (d) return d + "d " + h + "h";
  if (h) return h + "h " + m + "m";
  if (m) return m + "m " + s + "s";
  return s + "s";
}

function kernelName(value) {
  return value === "na" ? "M5 fast path" : "Standard";
}

function boolFrom(status, key, fallback) {
  return status.controls && key in status.controls ? !!status.controls[key] : !!fallback;
}

function intensityFrom(status) {
  return (status.controls && status.controls.intensity) || status.intensity || "full";
}

function updatePending(status) {
  if (!pending) return;
  let done = false;
  if (pending.kind === "paused") done = String(status.state || "").startsWith("paused") === pending.value;
  else if (pending.kind === "intensity") done = intensityFrom(status) === pending.value;
  else if (pending.kind === "only_ac") done = boolFrom(status, "only_ac", status.only_ac) === pending.value;
  else if (pending.kind === "start_at_login") done = boolFrom(status, "start_at_login", status.start_at_login) === pending.value;
  if (done) pending = null;
  if (pending) {
    $("applying").textContent = pending.label;
    if (pending.kind === "paused") {
      $("pauseButton").textContent = pending.label;
      $("pauseButton").disabled = true;
    }
  } else {
    $("applying").textContent = "";
    $("pauseButton").disabled = false;
  }
}

function renderStatus(status) {
  lastStatus = status || {};
  if (lastStatus.state === "uninstalled") showRemoved();
  const key = lastStatus.state || "checking";
  const pair = stateLabels[key] || stateLabels.checking;
  $("headline").textContent = {
    mining: "This Mac is mining Pearl.", checking: "Getting ready to mine…", startup: "Getting ready to mine…",
    paused: "Mining is paused.", paused_battery: "Paused while on battery.",
    paused_thermal: "Paused while your Mac cools down.", error: "Mining has stopped."
  }[key] || "This Mac is mining Pearl.";
  const label = key === "error" && lastStatus.last_error ? "Error: " + lastStatus.last_error : pair[0];
  $("statusPill").className = "pill" + (pair[1] ? " " + pair[1] : "");
  text("statusPill", label);
  if (key === "mining") {
    const tops = Number(lastStatus.tops || 0);
    $("speed").className = "v";
    text("speed", tops > 0 ? tops.toFixed(2) + " TOPS" : "Measuring…");
  } else if (key === "checking" || key === "startup") {
    $("speed").className = "v nospeed";
    text("speed", "Measuring…");
  } else {
    $("speed").className = "v nospeed";
    text("speed", "—");
  }
  text("accepted", lastStatus.shares_accepted || 0);
  text("rejected", lastStatus.shares_rejected || 0);
  text("uptime", seconds(lastStatus.uptime_s || lastStatus.uptime || 0));
  text("kernel", kernelName(lastStatus.kernel || lastStatus.effective_kernel));
  text("v4Ready", lastStatus.v4_ready ? "Yes" : "No");
  const intensity = intensityFrom(lastStatus);
  text("intensity", intensity.charAt(0).toUpperCase() + intensity.slice(1));
  const powerRaw = String(lastStatus.power_source || lastStatus.power || "");
  text("power", {ac: "Plugged in", battery: "On battery", desktop: "Plugged in"}[powerRaw] || "—");
  const paused = !!lastStatus.paused || key.startsWith("paused");
  $("pauseButton").textContent = paused ? "Resume" : "Pause";
  $("pauseSub").textContent = paused ? "Resume mining immediately." : "Pause mining immediately.";
  $("onlyAc").setAttribute("aria-checked", String(boolFrom(lastStatus, "only_ac", lastStatus.only_ac)));
  $("startAtLogin").setAttribute("aria-checked", String(boolFrom(lastStatus, "start_at_login", lastStatus.start_at_login)));
  for (const button of $("intensityGroup").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.intensity === intensity));
  }
  updatePending(lastStatus);
}

async function pollStatus() {
  if (removed) return;
  try {
    const response = await fetch("./api/status", {cache: "no-store"});
    if (!response.ok) throw new Error("Local status returned " + response.status);
    $("localError").textContent = "";
    const document = await response.json();
    const status = document.status || document;
    status.controls = document.controls || status.controls || {};
    renderStatus(status);
  } catch (error) {
    $("localError").textContent = removed ? "" : "Can't reach the miner on this Mac right now. If you just restarted, give it a minute.";
  }
}

async function control(body, waitLabel, pendingSpec) {
  pending = pendingSpec || null;
  $("applying").textContent = waitLabel;
  if (pending && pending.kind === "paused") {
    $("pauseButton").textContent = waitLabel;
    $("pauseButton").disabled = true;
  }
  $("localError").textContent = "";
  try {
    const response = await fetch("./api/control", {
      method: "POST",
      headers: {"Content-Type": "application/json", "X-Pearl-Local": PMK.localSecret},
      body: JSON.stringify(body)
    });
    if (!response.ok) throw new Error("Local control returned " + response.status);
    await response.json();
    if (!body.uninstall) await pollStatus();
    return true;
  } catch (error) {
    pending = null;
    $("applying").textContent = "";
    $("localError").textContent = "That change didn't go through. Refresh the page and try again.";
    $("pauseButton").disabled = false;
    renderStatus(lastStatus);
    return false;
  }
}

function showRemoved() {
  removed = true;
  $("app").classList.add("removed-mode");
  $("app").querySelectorAll("button").forEach((button) => { button.disabled = true; });
  $("applying").textContent = "Removed. You can close this page.";
}

async function fetchEarnings() {
  try {
    const url = "https://pearl.herominers.com/api/stats_address?address=" + encodeURIComponent(PMK.wallet);
    const response = await fetch(url, {cache: "no-store"});
    if (!response.ok) throw new Error("HeroMiners returned " + response.status);
    const data = await response.json();
    if (data.error || !data.stats) {
      $("earningsStatus").textContent = "No earnings yet. Shares show up here after your Mac finds its first one.";
      $("earningsBody").hidden = true;
      return;
    }
    $("earningsStatus").textContent = "For your whole wallet, including any other machines mining to it. From HeroMiners.";
    $("earningsBody").hidden = false;
    const stats = data.stats || data;
    const pendingBalance = Number(stats.balance || stats.pending || data.balance || 0);
    const paid = Number(stats.paid || stats.total_paid || data.paid || 0);
    text("pendingBalance", prl(pendingBalance));
    text("totalPaid", prl(paid));
    text("poolHashrate", rate(stats.hashrate));
    $("payoutProgress").style.width = Math.max(0, Math.min(100, pendingBalance / 1e8 * 100)) + "%";
    renderPayments(data.payments || stats.payments || []);
  } catch (error) {
    $("earningsStatus").textContent = "Earnings are unavailable right now.";
  }
}

function rate(value) {
  let n = Number(value) || 0;
  const units = ["H/s", "kH/s", "MH/s", "GH/s", "TH/s"];
  let index = 0;
  while (n >= 1000 && index < units.length - 1) { n /= 1000; index += 1; }
  return (index ? n.toFixed(2) : String(Math.round(n))) + " " + units[index];
}

function renderPayments(payments) {
  const rows = [];
  for (let i = 0; i < payments.length; i += 2) {
    const spec = String(payments[i] || "");
    const when = Number(payments[i + 1] || 0);
    const parts = spec.split(":");
    if (!parts[0]) continue;
    const tr = document.createElement("tr");
    const tx = document.createElement("td");
    tx.className = "tx";
    tx.textContent = parts[0].length > 16 ? parts[0].slice(0, 8) + "…" + parts[0].slice(-6) : parts[0];
    tx.title = parts[0];
    const amount = document.createElement("td");
    amount.className = "num";
    amount.textContent = prl(parts[1] || 0);
    const date = document.createElement("td");
    date.className = "num";
    date.textContent = when ? new Date(when * 1000).toLocaleDateString() : "-";
    tr.append(tx, amount, date);
    rows.push(tr);
  }
  const body = $("payments");
  body.replaceChildren(...rows);
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 3;
    td.textContent = "No payouts yet.";
    tr.append(td);
    body.append(tr);
  }
}

$("computerName").textContent = PMK.computerName;
$("heroLink").href = "https://pearl.herominers.com/";
$("pauseButton").addEventListener("click", () => {
  const next = !(!!lastStatus.paused || String(lastStatus.state || "").startsWith("paused"));
  control({paused: next}, next ? "Pausing…" : "Resuming…", {kind: "paused", value: next, label: next ? "Pausing…" : "Resuming…"});
});
$("intensityGroup").addEventListener("click", (event) => {
  if (!event.target.dataset.intensity) return;
  const value = event.target.dataset.intensity;
  control({intensity: value}, "Changing…", {kind: "intensity", value, label: "Changing…"});
});
$("onlyAc").addEventListener("click", () => {
  const value = $("onlyAc").getAttribute("aria-checked") !== "true";
  control({only_ac: value}, "Saving…", {kind: "only_ac", value, label: "Saving…"});
});
$("startAtLogin").addEventListener("click", () => {
  const value = $("startAtLogin").getAttribute("aria-checked") !== "true";
  control({start_at_login: value}, "Saving…", {kind: "start_at_login", value, label: "Saving…"});
});
$("removeButton").addEventListener("click", () => { $("removeConfirm").hidden = false; });
$("cancelRemove").addEventListener("click", () => { $("removeConfirm").hidden = true; });
$("confirmRemove").addEventListener("click", async () => {
  if (await control({uninstall: true}, "Removing...", null)) showRemoved();
});

renderStatus({state: "checking", controls: {intensity: "full", only_ac: true, start_at_login: true}});
pollStatus();
fetchEarnings();
setInterval(pollStatus, 2000);
setInterval(fetchEarnings, 60000);
</script>
</body>
</html>
"""


def _script_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
    )


def render_control_page(wallet: str, local_secret: str, port: int, computer_name: str) -> str:
    """Return the local control page HTML.

    The page intentionally keeps all agent calls relative to the secret URL path.
    The local HTTP handler enforces Host, path-secret, and POST-header checks.
    """

    config = {
        "wallet": wallet,
        "localSecret": local_secret,
        "port": int(port),
        "computerName": computer_name,
    }
    return (
        _HTML.replace("__PMK_CSS__", _CSS)
        .replace("__PMK_CONFIG__", _script_json(config))
    )
