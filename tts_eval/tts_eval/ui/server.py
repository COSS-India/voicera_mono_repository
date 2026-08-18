"""Lightweight UI: browse runs, read reports, play audio, compare two runs.

stdlib `http.server` only — no Flask, no Jinja, no npm build. A benchmark UI is a
convenience for a handful of reviewers, not a product surface; a dependency-free
server that starts with `python -m tts_eval.ui` is worth far more here than a
framework nobody on the team needs to install to look at a report.

Routes:
    GET /                                  run list
    GET /run/<run_id>                      run report (same renderer as the file)
    GET /run/<run_id>/run.json             source-of-truth record (download)
    GET /run/<run_id>/report.md            Markdown report (download)
    GET /run/<run_id>/report.html          standalone HTML report (download)
    GET /run/<run_id>/utterances.csv       CSV export
    GET /run/<run_id>/aggregates.csv
    GET /run/<run_id>/coverage.csv
    GET /audio/<run_id>/<filename>         WAV playback
    GET /compare?baseline=<id>&candidate=<id>   comparison report
    GET /static/style.css                  shared CSS
    GET /configs                           model card + suite config editor
    GET /launch                            run launcher with live progress
    GET /api/models                        list model card names (JSON)
    GET /api/suites                        list suite names (JSON)
    GET /api/model/<name>                  read model card YAML (JSON)
    GET /api/suite/<name>                  read suite YAML (JSON)
    GET /api/status                        current run progress (JSON)
    POST /api/model/<name>                 save model card YAML
    POST /api/suite/<name>                 save suite YAML
    POST /api/launch                       start evaluation run
    POST /api/runs/<run_id>/delete         delete one run (irreversible)
    POST /api/runs/clear                   delete every run (irreversible)

Most routes are read-only GET; the mutating POST routes (launch, config saves,
clear-all) exist for local/internal reviewer convenience, not multi-user
safety — there is no auth or CSRF surface. It is not hardened for exposure on
an untrusted network — see ``serve()``'s docstring.
"""
from __future__ import annotations

import html as _html
import json
import posixpath
import socket
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..compare import compare_runs
from ..config import MODELS_DIR, SUITES_DIR, list_model_cards, list_suites
from ..errors import StoreError
from ..store import RunStore, RunSummary
from .. import __version__
from ..report.csv_export import aggregates_csv, coverage_csv, utterances_csv
from ..report.html import _page  # shared page chrome, kept private to report.html
from ..report.html import render_comparison_html, render_run_html
from ..report.markdown import render_run_markdown
from ..report.style import CSS

# Per-run archival downloads served at /run/<id>/<name>, mapping the requested
# name to (content-type, download-filename-suffix). CSVs have their own route.
_RUN_DOWNLOADS: dict[str, tuple[str, str]] = {
    "run.json": ("application/json; charset=utf-8", "run.json"),
    "report.md": ("text/markdown; charset=utf-8", "report.md"),
    "report.html": ("text/html; charset=utf-8", "report.html"),
}

# ---------------------------------------------------------------------------
# Background run state
# ---------------------------------------------------------------------------
_run_state: dict = {
    "running": False,
    "model": "",
    "suite": "",
    "completed": 0,
    "total": 0,
    "last_id": "",
    "last_result": None,
    "run_id": None,
}
_run_lock = threading.Lock()


def _e(text: object) -> str:
    return _html.escape(str(text))


def _nav(*, active: str = "") -> str:
    def link(href: str, label: str, key: str) -> str:
        cls = ' class="active"' if key == active else ""
        return f'<a href="{href}"{cls}>{label}</a>'

    return (
        link("/", "Runs", "runs")
        + link("/configs", "Configs", "configs")
        + link("/launch", "Launch", "launch")
        + link("/compare", "Compare", "compare")
    )


def _run_option(summary: RunSummary, selected: str) -> str:
    sel = " selected" if summary.run_id == selected else ""
    label = f"{summary.display_name} — {summary.run_id} ({summary.created_at[:19]})"
    return f'<option value="{_e(summary.run_id)}"{sel}>{_e(label)}</option>'


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def _success_band(rate: float | None) -> str:
    """Map a success rate to a badge colour band, matching report.html thresholds."""
    if rate is None:
        return "neutral"
    if rate >= 0.99:
        return "good"
    if rate >= 0.9:
        return "warn"
    return "bad"


def render_index(store: RunStore) -> str:
    summaries = store.list_runs(limit=500)
    if not summaries:
        body = (
            f'<div class="topbar"><h1>tts_eval</h1><nav>{_nav(active="runs")}</nav></div>'
            f'<p class="muted">No runs found under <code>{_e(store.root)}</code>. '
            f'Start one from the <a href="/launch">Launch</a> tab.</p>'
        )
        return _page("tts_eval — runs", body)

    items = []
    for summary in summaries:
        success = f"{summary.success_rate:.0%}" if summary.success_rate is not None else "—"
        band = _success_band(summary.success_rate)
        # data-search backs the client-side filter box: one lowercase haystack so
        # a reviewer scanning dozens of runs can jump to one by name, label or id.
        haystack = _e(
            f"{summary.display_name} {summary.label} {summary.run_id}".lower()
        )
        items.append(
            f'<li data-search="{haystack}"><a class="run-link" href="/run/{_e(summary.run_id)}">'
            f'<div class="run-main">'
            f'<div class="title">{_e(summary.display_name)} — {_e(summary.label)}</div>'
            f'<div class="sub">{_e(summary.run_id)} &middot; {_e(summary.created_at[:19])} '
            f"&middot; concurrency {summary.concurrency}</div>"
            f'</div>'
            f'<div class="run-stats">'
            f'<span class="run-count">{summary.n_ok}/{summary.n_utterances}</span>'
            f'<span class="badge {band}">{success}</span>'
            f"</div></a>"
            f'<button class="run-del" title="Delete this run" aria-label="Delete run"'
            f' onclick="return deleteRun(event, \'{_e(summary.run_id)}\')">Delete</button>'
            f"</li>"
        )

    body = f"""
<div class="topbar"><h1>tts_eval</h1><nav>{_nav(active="runs")}</nav></div>
<p class="subtitle">{len(summaries)} run(s) under <code>{_e(store.root)}</code>
  &middot; <a href="#" class="small" onclick="return clearAllRuns()">Clear all runs</a></p>
<input class="run-filter" id="run-filter" type="search" autocomplete="off"
  placeholder="Filter runs by name, label or id…" aria-label="Filter runs">
<ul class="run-list" id="run-list">{''.join(items)}</ul>
<p class="run-empty" id="run-empty" hidden>No runs match that filter.</p>
<script>
(function () {{
  const input = document.getElementById('run-filter');
  const items = Array.from(document.querySelectorAll('#run-list > li'));
  const empty = document.getElementById('run-empty');
  input.addEventListener('input', function () {{
    const q = input.value.trim().toLowerCase();
    let shown = 0;
    for (const li of items) {{
      const match = !q || li.dataset.search.includes(q);
      li.hidden = !match;
      if (match) shown++;
    }}
    empty.hidden = shown > 0;
  }});
}})();

async function clearAllRuns() {{
  const n = {len(summaries)};
  if (!confirm('Delete all ' + n + ' run(s) permanently? This cannot be undone.')) return false;
  if (!confirm('Really sure? Audio and reports for every run will be deleted from disk.')) return false;
  try {{
    const r = await fetch('/api/runs/clear', {{ method: 'POST' }});
    const d = await r.json();
    if (d.status === 'ok') {{ location.reload(); }}
    else {{ alert(d.message || 'Failed to clear runs'); }}
  }} catch(e) {{ alert('Failed to clear runs: ' + e); }}
  return false;
}}

async function deleteRun(ev, id) {{
  // The button sits outside the run's <a>, so there is nothing to navigate,
  // but stop the event anyway in case the markup changes later.
  ev.preventDefault(); ev.stopPropagation();
  if (!confirm('Delete run ' + id + ' permanently? Its audio and report are removed from disk.')) return false;
  try {{
    const r = await fetch('/api/runs/' + encodeURIComponent(id) + '/delete', {{ method: 'POST' }});
    const d = await r.json();
    if (d.status === 'ok') {{ location.reload(); }}
    else {{ alert(d.message || 'Failed to delete run'); }}
  }} catch(e) {{ alert('Failed to delete run: ' + e); }}
  return false;
}}
</script>
"""
    return _page("tts_eval — runs", body)


def render_compare_form(store: RunStore, *, baseline: str = "", candidate: str = "") -> str:
    summaries = store.list_runs(limit=500)
    baseline_options = "".join(_run_option(s, baseline) for s in summaries)
    candidate_options = "".join(_run_option(s, candidate) for s in summaries)
    body = f"""
<div class="topbar"><h1>Compare runs</h1><nav>{_nav(active="compare")}</nav></div>
<form class="compare-form" method="get" action="/compare">
  <label>Baseline
    <select name="baseline" required>
      <option value="" disabled {'selected' if not baseline else ''}>choose a run…</option>
      {baseline_options}
    </select>
  </label>
  <label>Candidate
    <select name="candidate" required>
      <option value="" disabled {'selected' if not candidate else ''}>choose a run…</option>
      {candidate_options}
    </select>
  </label>
  <button type="submit">Compare</button>
</form>
<p class="small muted">Comparison pairs utterances by id and requires both runs to share
a dataset and concurrency — see the comparability checks in the result.</p>
"""
    return _page("tts_eval — compare", body)


# Starter templates shown in the "new config" dialog: a single commented YAML the
# reviewer edits in place, rather than a field-per-input form. Comments carry the
# structure so a newcomer learns the shape from the sample itself.
_MODEL_SAMPLE = """# Model card — how to reach one TTS system. Edit the values below.
model_id: my-tts-model
model_version: "1.0"
provider: MyTeam

# Transport adapter. Built-in: http_rest, websocket_pcm, mock, replay.
adapter: http_rest
adapter_config:
  url: ${MY_MODEL_URL:-http://localhost:8000/synthesize}
  method: POST
  # A JSON API returning base64 audio? Set response_format: json_base64.

sample_rate: 24000
voices: [default]
default_voice: default
languages: [en]

# deterministic | best_effort  (recorded in the run; runs across a mismatch are refused).
determinism: best_effort
"""

_SUITE_SAMPLE = """# Suite — the fixed protocol every model in a bake-off runs unchanged.
suite_id: my-suite
dataset: indic_conversational_v1

# Metric tier: core | standard | all  (all adds the optional perceptual backends).
metrics: standard

# Protocol, not a speed knob: runs at a different concurrency are non-comparable.
concurrency: 1
seed: 1234
save_audio: true
"""


def render_configs_page() -> str:
    """Configuration manager: list and edit model cards and suite configs."""
    models = list_model_cards()
    suites = list_suites()

    model_items = "".join(
        f'<li onclick="loadConfig(\'model\',\'{_e(m)}\')" id="item-model-{_e(m)}">{_e(m)}</li>'
        for m in models
    ) or '<li class="muted">No model cards found</li>'

    suite_items = "".join(
        f'<li onclick="loadConfig(\'suite\',\'{_e(s)}\')" id="item-suite-{_e(s)}">{_e(s)}</li>'
        for s in suites
    ) or '<li class="muted">No suites found</li>'

    model_sample = _e(_MODEL_SAMPLE)
    suite_sample = _e(_SUITE_SAMPLE)

    body = f"""
<div class="topbar"><h1>Configurations</h1><nav>{_nav(active="configs")}</nav></div>
<p class="subtitle">Edit model cards and evaluation suites. Changes are saved to <code>{_e(MODELS_DIR.parent)}</code>.</p>

<div class="config-grid">
  <div class="config-sidebar">
    <div class="config-section">
      <h3>Model Cards</h3>
      <ul class="config-list" id="model-list">{model_items}</ul>
      <button class="btn btn-secondary" style="width:100%;margin-top:.5rem"
        onclick="document.getElementById('model-dialog').showModal()">+ New Model Card</button>
    </div>
    <div class="config-section">
      <h3>Suites</h3>
      <ul class="config-list" id="suite-list">{suite_items}</ul>
      <button class="btn btn-secondary" style="width:100%;margin-top:.5rem"
        onclick="document.getElementById('suite-dialog').showModal()">+ New Suite</button>
    </div>
  </div>

  <div>
    <div class="editor-toolbar">
      <span class="editor-title" id="editor-title">Select a config to edit</span>
      <button class="btn" onclick="saveConfig()" id="save-btn" disabled>Save</button>
    </div>
    <div class="editor-wrap">
      <pre class="editor-highlight" id="editor-highlight" aria-hidden="true"></pre>
      <textarea class="editor-area" id="editor" spellcheck="false"
        placeholder="Select a model card or suite from the sidebar, or create a new one, to view and edit its YAML." disabled></textarea>
    </div>
  </div>
</div>

<dialog class="modal" id="model-dialog">
  <form method="dialog" onsubmit="return createModel(event)">
    <div class="modal-head">
      <div><h3>New Model Card</h3><p class="modal-hint">Name it, then edit the sample YAML. Saved to the editor as a draft.</p></div>
    </div>
    <div class="modal-body">
      <div class="form-group">
        <label>Name (file name)</label>
        <input id="md-name" placeholder="my-tts-model" required
          pattern="[A-Za-z0-9_-]+" title="Letters, digits, - and _ only">
      </div>
      <div class="form-group">
        <label>YAML</label>
        <textarea class="yaml-input" id="md-yaml" spellcheck="false">{model_sample}</textarea>
      </div>
    </div>
    <div class="modal-foot">
      <button type="button" class="btn btn-secondary" onclick="this.closest('dialog').close()">Cancel</button>
      <button type="submit" class="btn">Create</button>
    </div>
  </form>
</dialog>

<dialog class="modal" id="suite-dialog">
  <form method="dialog" onsubmit="return createSuite(event)">
    <div class="modal-head">
      <div><h3>New Suite</h3><p class="modal-hint">Name it, then edit the sample YAML. Saved to the editor as a draft.</p></div>
    </div>
    <div class="modal-body">
      <div class="form-group">
        <label>Name (file name)</label>
        <input id="su-name" placeholder="my-suite" required
          pattern="[A-Za-z0-9_-]+" title="Letters, digits, - and _ only">
      </div>
      <div class="form-group">
        <label>YAML</label>
        <textarea class="yaml-input" id="su-yaml" spellcheck="false">{suite_sample}</textarea>
      </div>
    </div>
    <div class="modal-foot">
      <button type="button" class="btn btn-secondary" onclick="this.closest('dialog').close()">Cancel</button>
      <button type="submit" class="btn">Create</button>
    </div>
  </form>
</dialog>

<script>
let curType = null, curName = null;

// -- YAML syntax highlighting --------------------------------------------
// A colored <pre> sits behind the transparent-text <textarea> and mirrors
// its content + scroll position; the textarea keeps native editing,
// selection and caret behaviour, so no editor library is needed.
const YAML_TOKEN_RE =
  /(^\\s*-\\s)|(^\\s*[\\w.-]+\\s*:)(?=\\s|$)|(#.*$)|("(?:[^"\\\\]|\\\\.)*"|'(?:[^'\\\\]|\\\\.)*')|(\\$\\{{[^}}]*\\}})|\\b(true|false|null|yes|no|True|False|Null|Yes|No|~)\\b|\\b(-?\\d+(?:\\.\\d+)?)\\b/g;

function escapeHtml(s) {{
  return s.replace(/[&<>"]/g, c => ({{'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}})[c]);
}}

function highlightYamlLine(line) {{
  let out = '', last = 0, m;
  YAML_TOKEN_RE.lastIndex = 0;
  while ((m = YAML_TOKEN_RE.exec(line)) !== null) {{
    if (m.index > last) out += escapeHtml(line.slice(last, m.index));
    const cls = m[2] ? 'yaml-key' : m[3] ? 'yaml-comment' : m[4] ? 'yaml-string'
      : m[5] ? 'yaml-env' : m[6] ? 'yaml-bool' : m[7] ? 'yaml-number' : 'yaml-dash';
    out += '<span class="' + cls + '">' + escapeHtml(m[0]) + '</span>';
    last = m.index + m[0].length;
  }}
  return out + escapeHtml(line.slice(last));
}}

function updateHighlight() {{
  const text = document.getElementById('editor').value;
  document.getElementById('editor-highlight').innerHTML =
    text.split('\\n').map(highlightYamlLine).join('\\n') + '\\n';
}}

function syncEditorScroll() {{
  const editor = document.getElementById('editor');
  const hi = document.getElementById('editor-highlight');
  hi.scrollTop = editor.scrollTop;
  hi.scrollLeft = editor.scrollLeft;
}}

document.getElementById('editor').addEventListener('input', updateHighlight);
document.getElementById('editor').addEventListener('scroll', syncEditorScroll);

function setActive(type, name) {{
  document.querySelectorAll('.config-list li').forEach(li => li.classList.remove('active'));
  const el = document.getElementById('item-' + type + '-' + name);
  if (el) el.classList.add('active');
}}

async function loadConfig(type, name) {{
  try {{
    const r = await fetch('/api/' + type + '/' + encodeURIComponent(name));
    const d = await r.json();
    if (d.status === 'error') {{ toast(d.message, true); return; }}
    document.getElementById('editor').value = d.content;
    document.getElementById('editor').disabled = false;
    document.getElementById('save-btn').disabled = false;
    document.getElementById('editor-title').textContent = name + '.yaml (' + type + ')';
    curType = type; curName = name;
    setActive(type, name);
    updateHighlight();
  }} catch(e) {{ toast('Failed to load: ' + e, true); }}
}}

async function saveConfig() {{
  if (!curType || !curName) return;
  const content = document.getElementById('editor').value;
  try {{
    const r = await fetch('/api/' + curType + '/' + encodeURIComponent(curName), {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{content}})
    }});
    const d = await r.json();
    if (d.status === 'ok') toast('Saved ' + curName + '.yaml');
    else toast(d.message, true);
  }} catch(e) {{ toast('Save failed: ' + e, true); }}
}}

// -- "new config" dialogs ------------------------------------------------
// One name + one YAML textarea prefilled with a commented sample, so the
// structure teaches itself. On create the YAML drops into the editor as an
// unsaved draft; the reviewer clicks Save to write it.
function openInEditor(type, name, yaml) {{
  const ed = document.getElementById('editor');
  ed.value = yaml;
  ed.disabled = false;
  document.getElementById('save-btn').disabled = false;
  document.getElementById('editor-title').textContent = name + '.yaml (' + type + ') — new draft, click Save';
  curType = type; curName = name;
  updateHighlight();
  ed.focus();
  toast('Draft ready — review and click Save');
}}

function createConfig(ev, type, nameId, yamlId, dialogId) {{
  ev.preventDefault();
  const name = document.getElementById(nameId).value.trim();
  if (!/^[A-Za-z0-9_-]+$/.test(name)) {{ toast('Name: letters, digits, - and _ only', true); return false; }}
  const yaml = document.getElementById(yamlId).value;
  document.getElementById(dialogId).close();
  openInEditor(type, name, yaml);
  return false;
}}

function createModel(ev) {{ return createConfig(ev, 'model', 'md-name', 'md-yaml', 'model-dialog'); }}
function createSuite(ev) {{ return createConfig(ev, 'suite', 'su-name', 'su-yaml', 'suite-dialog'); }}

function toast(msg, err) {{
  const el = document.createElement('div');
  el.className = 'toast ' + (err ? 'toast-err' : 'toast-ok');
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 2600);
}}
</script>
"""
    return _page("tts_eval — configs", body, inline_css=False)


def render_launch_page() -> str:
    """Run launcher with live progress tracking."""
    models = list_model_cards()
    suites = list_suites()

    model_opts = "".join(f'<option value="{_e(m)}">{_e(m)}</option>' for m in models)
    suite_opts = "".join(f'<option value="{_e(s)}">{_e(s)}</option>' for s in suites)

    body = f"""
<div class="topbar"><h1>Launch Evaluation</h1><nav>{_nav(active="launch")}</nav></div>
<p class="subtitle">Select a model and suite, then launch an evaluation run.</p>

<div class="launch-form" style="max-width:36rem">
  <div class="form-group">
    <label>Model Card</label>
    <select id="model-select">
      <option value="" disabled selected>Select model…</option>
      {model_opts}
    </select>
  </div>
  <div class="form-group">
    <label>Suite</label>
    <select id="suite-select">
      <option value="" disabled selected>Select suite…</option>
      {suite_opts}
    </select>
  </div>
  <div class="form-group">
    <label>Label (optional)</label>
    <input id="label-input" placeholder="e.g. experiment_v2">
  </div>
  <button class="btn" id="launch-btn" onclick="launchRun()">Launch Run</button>
</div>

<div class="status-box" id="status-box">
  <span class="muted">No run in progress. Select a model and suite, then click Launch.</span>
</div>

<script>
let poll = null;

async function launchRun() {{
  const model = document.getElementById('model-select').value;
  const suite = document.getElementById('suite-select').value;
  const label = document.getElementById('label-input').value;
  if (!model || !suite) {{ alert('Select both model and suite.'); return; }}
  document.getElementById('launch-btn').disabled = true;
  document.getElementById('status-box').innerHTML = '<span class="muted">Starting…</span>';
  try {{
    const r = await fetch('/api/launch', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{model, suite, label}})
    }});
    const d = await r.json();
    if (d.status === 'error') {{
      document.getElementById('status-box').innerHTML = '<span class="badge bad">Error</span> ' + d.message;
      document.getElementById('launch-btn').disabled = false;
      return;
    }}
    poll = setInterval(checkStatus, 1000);
    checkStatus();
  }} catch(e) {{
    document.getElementById('status-box').innerHTML = '<span class="badge bad">Error</span> ' + e;
    document.getElementById('launch-btn').disabled = false;
  }}
}}

async function checkStatus() {{
  try {{
    const r = await fetch('/api/status');
    const d = await r.json();
    const box = document.getElementById('status-box');
    if (d.running) {{
      const pct = d.total > 0 ? (d.completed / d.total * 100).toFixed(1) : 0;
      box.innerHTML = '<strong>Running:</strong> ' + d.model + ' / ' + d.suite + '<br>'
        + d.completed + '/' + d.total + ' utterances'
        + '<div class="progress-bar"><div class="progress-fill" style="width:' + pct + '%"></div></div>'
        + '<span class="muted">Last: ' + (d.last_id || '—') + '</span>';
    }} else {{
      clearInterval(poll);
      document.getElementById('launch-btn').disabled = false;
      if (d.last_result === 'success') {{
        box.innerHTML = '<span class="badge good">Complete</span> '
          + 'Run <a href="/run/' + d.run_id + '">' + d.run_id + '</a> — '
          + '<a href="/run/' + d.run_id + '">View Report</a>';
      }} else if (d.last_result) {{
        box.innerHTML = '<span class="badge bad">Failed</span> ' + d.last_result;
      }} else {{
        box.innerHTML = '<span class="muted">No run in progress.</span>';
      }}
    }}
  }} catch(e) {{ /* ignore transient fetch errors */ }}
}}

// Check on page load if a run is already going
checkStatus();
</script>
"""
    return _page("tts_eval — launch", body, inline_css=False)


# ---------------------------------------------------------------------------
# Background run execution
# ---------------------------------------------------------------------------

def _run_eval_background(model_name: str, suite_name: str, label: str, store: RunStore) -> None:
    """Execute an evaluation in a background thread."""
    try:
        from ..config import load_model_card, load_suite
        from ..runner import build_plan, run_sync
        from ..report import write_run_report

        card = load_model_card(model_name)
        suite = load_suite(suite_name)
        plan = build_plan(card, suite, output_dir=store.root, label=label or None)

        with _run_lock:
            _run_state["total"] = len(plan.dataset)

        def progress(completed: int, total: int, last_id: str) -> None:
            with _run_lock:
                _run_state["completed"] = completed
                _run_state["total"] = total
                _run_state["last_id"] = last_id

        record = run_sync(plan, progress=progress)
        run_dir = store.save(record)
        write_run_report(record, run_dir)

        with _run_lock:
            _run_state["running"] = False
            _run_state["last_result"] = "success"
            _run_state["run_id"] = record.run_id

    except Exception as exc:
        with _run_lock:
            _run_state["running"] = False
            _run_state["last_result"] = str(exc)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    """One handler class per server instance, via a closure in :func:`make_app`.

    ``store`` is bound per-instance rather than read from a global so tests can
    spin up multiple servers against different stores in one process.
    """

    store: RunStore
    server_version = f"tts_eval/{__version__}"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        # Default logs to stderr with no level distinction; route through a single
        # line so a reviewer running this in a terminal is not drowned in noise.
        pass

    # -- helpers ------------------------------------------------------------
    def _send(
        self, status: HTTPStatus, body: bytes, content_type: str, *, download: str | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Audio and CSV are static once a run finishes; HTML is cheap to
        # regenerate and always reflects the current run.json, so it is not cached.
        self.send_header("Cache-Control", "no-store")
        if download is not None:
            # Force a "save as" with a stable filename rather than rendering the
            # artefact inline in the browser tab.
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_html(self, status: HTTPStatus, document: str) -> None:
        self._send(status, document.encode("utf-8"), "text/html; charset=utf-8")

    def _send_text(self, status: HTTPStatus, text: str, content_type: str) -> None:
        self._send(status, text.encode("utf-8"), content_type)

    def _send_json(self, status: HTTPStatus, obj: dict) -> None:
        data = json.dumps(obj, default=str).encode("utf-8")
        self._send(status, data, "application/json")

    def _error_page(self, status: HTTPStatus, message: str) -> None:
        body = (
            f'<div class="topbar"><h1>{status.value} {_e(status.phrase)}</h1>'
            f'<nav>{_nav()}</nav></div><p>{_e(message)}</p>'
        )
        self._send_html(status, _page(f"{status.value} {status.phrase}", body))

    @staticmethod
    def _safe_name(raw: str) -> str | None:
        """Reject path traversal in a URL path segment.

        Every route below interpolates a run id or filename straight into a
        filesystem path; without this check ``/audio/../../etc/passwd``-style
        segments would need catching in three places instead of one.
        """
        name = urllib.parse.unquote(raw)
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            return None
        # normpath catches encoded traversal (e.g. "%2e%2e") that unquote alone
        # would pass through as a literal ".." string, which the check above
        # already rejects — this is defence in depth for odd encodings.
        if posixpath.normpath(name) != name:
            return None
        return name

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    # -- routing --------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        try:
            self._route_get()
        except StoreError as e:
            self._error_page(HTTPStatus.NOT_FOUND, str(e))
        except BrokenPipeError:
            pass  # client disconnected mid-response; nothing to report

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib method name
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._route_post()
        except BrokenPipeError:
            pass

    def _route_get(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "":
            self._send_html(HTTPStatus.OK, render_index(self.store))
            return

        if path == "/static/style.css":
            self._send_text(HTTPStatus.OK, CSS, "text/css; charset=utf-8")
            return

        if path == "/configs":
            self._send_html(HTTPStatus.OK, render_configs_page())
            return

        if path == "/launch":
            self._send_html(HTTPStatus.OK, render_launch_page())
            return

        if path == "/compare":
            baseline = (query.get("baseline") or [""])[0]
            candidate = (query.get("candidate") or [""])[0]
            if not baseline or not candidate:
                self._send_html(HTTPStatus.OK, render_compare_form(self.store))
                return
            self._handle_compare(baseline, candidate)
            return

        segments = [s for s in path.split("/") if s]

        # API routes (GET)
        if len(segments) == 2 and segments[0] == "api":
            if segments[1] == "models":
                self._send_json(HTTPStatus.OK, {"models": list_model_cards()})
                return
            if segments[1] == "suites":
                self._send_json(HTTPStatus.OK, {"suites": list_suites()})
                return
            if segments[1] == "status":
                with _run_lock:
                    self._send_json(HTTPStatus.OK, dict(_run_state))
                return

        if len(segments) == 3 and segments[0] == "api":
            if segments[1] == "model":
                self._api_read_config("model", segments[2])
                return
            if segments[1] == "suite":
                self._api_read_config("suite", segments[2])
                return

        if len(segments) == 2 and segments[0] == "audio":
            self._handle_audio(segments[1], query)
            return
        if len(segments) == 3 and segments[0] == "audio":
            self._handle_audio_file(segments[1], segments[2])
            return
        if len(segments) == 2 and segments[0] == "run" and segments[1].endswith(".csv"):
            # unreachable pattern guard; real csv routes are 3 segments (see below)
            pass
        if len(segments) == 2 and segments[0] == "run":
            self._handle_run(segments[1])
            return
        if len(segments) == 3 and segments[0] == "run" and segments[2].endswith(".csv"):
            self._handle_run_csv(segments[1], segments[2])
            return
        if len(segments) == 3 and segments[0] == "run" and segments[2] in _RUN_DOWNLOADS:
            self._handle_run_download(segments[1], segments[2])
            return

        self._error_page(HTTPStatus.NOT_FOUND, f"No route for {path}")

    def _route_post(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        segments = [s for s in path.split("/") if s]

        if len(segments) == 3 and segments[0] == "api":
            if segments[1] == "model":
                self._api_write_config("model", segments[2])
                return
            if segments[1] == "suite":
                self._api_write_config("suite", segments[2])
                return

        if len(segments) == 2 and segments[0] == "api" and segments[1] == "launch":
            self._api_launch()
            return

        if len(segments) == 3 and segments[0] == "api" and segments[1] == "runs" and segments[2] == "clear":
            self._api_clear_runs()
            return

        if (
            len(segments) == 4
            and segments[0] == "api"
            and segments[1] == "runs"
            and segments[3] == "delete"
        ):
            self._api_delete_run(segments[2])
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": "unknown endpoint"})

    # -- API handlers -------------------------------------------------------
    def _api_read_config(self, config_type: str, name_raw: str) -> None:
        name = self._safe_name(name_raw)
        if name is None:
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": "invalid name"})
            return
        directory = MODELS_DIR if config_type == "model" else SUITES_DIR
        for ext in (".yaml", ".yml"):
            p = directory / f"{name}{ext}"
            if p.is_file():
                content = p.read_text(encoding="utf-8")
                self._send_json(HTTPStatus.OK, {"name": name, "type": config_type, "content": content})
                return
        self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": f"{config_type} '{name}' not found"})

    def _api_write_config(self, config_type: str, name_raw: str) -> None:
        name = self._safe_name(name_raw)
        if name is None:
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": "invalid name"})
            return
        try:
            body = json.loads(self._read_body().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": f"bad request: {exc}"})
            return

        content = body.get("content", "")
        if not content.strip():
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": "empty content"})
            return

        # Validate YAML
        try:
            import yaml
            yaml.safe_load(content)
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": f"invalid YAML: {exc}"})
            return

        directory = MODELS_DIR if config_type == "model" else SUITES_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.yaml"
        path.write_text(content, encoding="utf-8")
        self._send_json(HTTPStatus.OK, {"status": "ok", "path": str(path)})

    def _api_launch(self) -> None:
        try:
            body = json.loads(self._read_body().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": f"bad request: {exc}"})
            return

        model_name = body.get("model", "")
        suite_name = body.get("suite", "")
        label = body.get("label", "")

        if not model_name or not suite_name:
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": "model and suite required"})
            return

        with _run_lock:
            if _run_state["running"]:
                self._send_json(HTTPStatus.CONFLICT, {
                    "status": "error",
                    "message": f"Run already in progress: {_run_state['model']} / {_run_state['suite']}"
                })
                return
            _run_state.update({
                "running": True,
                "model": model_name,
                "suite": suite_name,
                "completed": 0,
                "total": 0,
                "last_id": "",
                "last_result": None,
                "run_id": None,
            })

        t = threading.Thread(
            target=_run_eval_background,
            args=(model_name, suite_name, label, self.store),
            daemon=True,
        )
        t.start()

        self._send_json(HTTPStatus.OK, {"status": "started", "message": f"Started {model_name} / {suite_name}"})

    def _api_clear_runs(self) -> None:
        with _run_lock:
            if _run_state["running"]:
                self._send_json(HTTPStatus.CONFLICT, {
                    "status": "error",
                    "message": "Cannot clear runs while an evaluation is in progress",
                })
                return

        count = self.store.clear_all()
        self._send_json(HTTPStatus.OK, {"status": "ok", "removed": count})

    def _api_delete_run(self, run_id_raw: str) -> None:
        run_id = self._safe_name(run_id_raw)
        if run_id is None:
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": "invalid run id"})
            return
        # Deleting the run that is still being written would race its save; refuse
        # while it is the one in progress. Other completed runs are fine to remove
        # mid-evaluation.
        with _run_lock:
            if _run_state["running"] and _run_state.get("run_id") == run_id:
                self._send_json(HTTPStatus.CONFLICT, {
                    "status": "error",
                    "message": "Cannot delete a run while it is still in progress",
                })
                return
        try:
            removed = self.store.delete(run_id)
        except StoreError as e:
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": str(e)})
            return
        if not removed:
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": f"run {run_id} not found"})
            return
        self._send_json(HTTPStatus.OK, {"status": "ok", "removed": run_id})

    # -- existing page handlers ---------------------------------------------
    def _handle_run(self, run_id_raw: str) -> None:
        run_id = self._safe_name(run_id_raw)
        if run_id is None:
            self._error_page(HTTPStatus.BAD_REQUEST, "invalid run id")
            return
        record = self.store.load(run_id)
        document = render_run_html(
            record,
            inline_css=False,
            audio_base=f"/audio/{record.run_id}/",
            nav_html=_nav(),
            export_base=f"/run/{record.run_id}/",
        )
        self._send_html(HTTPStatus.OK, document)

    def _handle_run_download(self, run_id_raw: str, filename: str) -> None:
        """Serve a run's archival artefacts as file downloads.

        JSON is streamed straight from the source-of-truth ``run.json``; the
        Markdown and standalone-HTML reports are rendered fresh from the loaded
        record so a download always reflects the current run.json (post-hoc
        subjective scores or sign-offs included), not a stale on-disk copy.
        """
        run_id = self._safe_name(run_id_raw)
        if run_id is None:
            self._error_page(HTTPStatus.BAD_REQUEST, "invalid run id")
            return
        record = self.store.load(run_id)
        content_type, disposition = _RUN_DOWNLOADS[filename]
        if filename == "run.json":
            body = self.store.path_for(record.run_id).read_bytes()
        elif filename == "report.md":
            body = render_run_markdown(record).encode("utf-8")
        else:  # report.html — standalone, inline CSS, audio via the server routes
            body = render_run_html(
                record, audio_base=f"/audio/{record.run_id}/"
            ).encode("utf-8")
        download_name = f"{record.run_id}-{disposition}"
        self._send(HTTPStatus.OK, body, content_type, download=download_name)

    def _handle_run_csv(self, run_id_raw: str, filename: str) -> None:
        run_id = self._safe_name(run_id_raw)
        exporters = {
            "utterances.csv": utterances_csv,
            "aggregates.csv": aggregates_csv,
            "coverage.csv": coverage_csv,
        }
        exporter = exporters.get(filename)
        if run_id is None or exporter is None:
            self._error_page(HTTPStatus.NOT_FOUND, "unknown export")
            return
        record = self.store.load(run_id)
        self._send_text(HTTPStatus.OK, exporter(record), "text/csv; charset=utf-8")

    def _handle_audio_file(self, run_id_raw: str, filename_raw: str) -> None:
        run_id = self._safe_name(run_id_raw)
        filename = self._safe_name(filename_raw)
        if run_id is None or filename is None or not filename.endswith(".wav"):
            self._error_page(HTTPStatus.BAD_REQUEST, "invalid audio path")
            return
        path = self.store.audio_dir(run_id) / filename
        if not path.is_file():
            self._error_page(HTTPStatus.NOT_FOUND, f"no audio file {filename} in run {run_id}")
            return
        self._send(HTTPStatus.OK, path.read_bytes(), "audio/wav")

    def _handle_audio(self, run_id_raw: str, query: dict) -> None:
        # A bare /audio/<run_id> with no filename is a navigation mistake, not a
        # missing file; point back at the run rather than a bare 404.
        run_id = self._safe_name(run_id_raw)
        if run_id is None:
            self._error_page(HTTPStatus.BAD_REQUEST, "invalid run id")
            return
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", f"/run/{urllib.parse.quote(run_id)}")
        self.end_headers()

    def _handle_compare(self, baseline_raw: str, candidate_raw: str) -> None:
        baseline_id = self._safe_name(baseline_raw)
        candidate_id = self._safe_name(candidate_raw)
        if baseline_id is None or candidate_id is None:
            self._error_page(HTTPStatus.BAD_REQUEST, "invalid run id")
            return
        baseline = self.store.load(baseline_id)
        candidate = self.store.load(candidate_id)
        comparison = compare_runs(baseline, candidate)
        document = render_comparison_html(comparison, inline_css=False, nav_html=_nav(active="compare"))
        self._send_html(HTTPStatus.OK, document)


def make_app(store: RunStore) -> type[Handler]:
    """Bind a store to a fresh Handler subclass, so multiple servers can coexist."""
    return type("BoundHandler", (Handler,), {"store": store})


def serve(
    root: str | Path = "runs", *, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    """Start the UI and return the running server (caller controls its lifetime).

    Binds to ``127.0.0.1`` by default: this UI has no authentication, and a run
    directory can contain audio of real user data plus API-adjacent metadata, so
    it is not meant to be reachable beyond the machine it runs on. Pass an
    explicit ``host`` only on a network you trust, and put a reverse proxy with
    auth in front of it if it must be reachable more broadly.
    """
    store = RunStore(root)
    httpd = ThreadingHTTPServer((host, port), make_app(store))
    return httpd


def serve_forever(root: str | Path = "runs", *, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Blocking entry point used by ``python -m tts_eval.ui``."""
    httpd = serve(root, host=host, port=port)
    actual_port = httpd.server_address[1]
    print(f"tts_eval UI: http://{host}:{actual_port}  (runs: {Path(root).resolve()})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def free_port() -> int:
    """Pick an unused port. Used by tests so parallel test runs cannot collide."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


__all__ = [
    "Handler", "free_port", "make_app", "render_compare_form", "render_configs_page",
    "render_index", "render_launch_page", "serve", "serve_forever",
]
