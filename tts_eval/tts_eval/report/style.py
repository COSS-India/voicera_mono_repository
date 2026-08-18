"""Shared CSS, as a Python string rather than a static file.

Two consumers need the same rules: a standalone HTML report must be a single
file that opens with no web server (so the CSS is inlined via ``<style>``), and
the lightweight UI serves it once at ``/static/style.css`` for every page.
Keeping one string avoids the two ever drifting into different colour
vocabularies for "good"/"warn"/"bad" — and it is also where every reusable
component (buttons, forms, progress bar, toast, config editor) lives, so
per-page inline ``<style>`` blocks in :mod:`tts_eval.ui.server` don't fork them.

Theme behavior:
    - Light mode is the default.
    - A page-level Light/Dark toggle is provided by ``report.html``.
    - The selected theme is persisted in ``localStorage`` when available.
    - No external JavaScript dependency is required, preserving the standalone
      HTML report contract.
"""
from __future__ import annotations

CSS = """
:root {
  color-scheme: light;

  --bg: #f3f4f7;
  --surface: #ffffff;
  --surface-subtle: #f6f7fa;
  --text: #1a1d23;
  --text-secondary: #545b66;
  --accent: #3b5fe0;
  --accent-hover: #2f4dc2;
  --accent-soft: #eaeefb;

  --good: #157f45;
  --good-bg: #e5f6ec;
  --warn: #92620a;
  --warn-bg: #fdf1dc;
  --bad: #c22a3e;
  --bad-bg: #fbe7ea;
  --border: #e1e4ea;
  --muted: #6b7280;

  --syntax-key: #1749c4;
  --syntax-string: #157f45;
  --syntax-number: #ae5d00;
  --syntax-comment: #7b8291;
  --syntax-env: #8933c9;
  --syntax-dash: #b5313f;

  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 14px;
  --shadow-sm: 0 1px 2px rgba(20, 24, 33, .06);
  --shadow-md: 0 8px 24px -8px rgba(20, 24, 33, .16);
}

html[data-theme="dark"] {
  color-scheme: dark;

  --bg: #0c0e12;
  --surface: #15171d;
  --surface-subtle: #1c1f27;
  --text: #e7e9ee;
  --text-secondary: #9aa1ad;
  --accent: #7b96ff;
  --accent-hover: #97acff;
  --accent-soft: #1e2740;

  --good: #4cc785;
  --good-bg: #123322;
  --warn: #e0a83c;
  --warn-bg: #332711;
  --bad: #f16d7c;
  --bad-bg: #391a20;
  --border: #2a2e38;
  --muted: #8b929e;

  --syntax-key: #7aa2f7;
  --syntax-string: #4cc785;
  --syntax-number: #e0a83c;
  --syntax-comment: #767d8a;
  --syntax-env: #c792ea;
  --syntax-dash: #f16d7c;
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, .4);
  --shadow-md: 0 8px 28px -8px rgba(0, 0, 0, .6);
}

* { box-sizing: border-box; }

html { background: var(--bg); }

body {
  font: 15px/1.6 -apple-system, "Segoe UI", system-ui, sans-serif;
  max-width: 68rem;
  margin: 0 auto;
  padding: 1.5rem 1.25rem 4rem;
  background: var(--bg);
  color: var(--text);
  -webkit-font-smoothing: antialiased;
}

h1 { font-size: 1.5rem; font-weight: 700; margin: 0 0 .2rem; letter-spacing: -.01em; }
h2 {
  font-size: 1.05rem;
  font-weight: 650;
  margin: 2rem 0 .85rem;
  padding-bottom: .5rem;
  border-bottom: 1px solid var(--border);
}
h3 { font-size: .95rem; font-weight: 650; margin: 1.25rem 0 .5rem; }

a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent-hover); text-decoration: underline; }

.subtitle {
  color: var(--text-secondary);
  margin: 0 0 1.5rem;
  font-size: .92rem;
}

/* -- layout: header card ------------------------------------------------- */

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: .75rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-sm);
  padding: 1rem 1.25rem;
  margin-bottom: 1.5rem;
  position: sticky;
  top: .75rem;
  z-index: 10;
}

.topbar nav {
  display: flex;
  align-items: center;
  gap: .25rem;
  flex-wrap: wrap;
}

.topbar nav a {
  padding: .4rem .75rem;
  border-radius: 999px;
  font-size: .86rem;
  font-weight: 550;
  color: var(--text-secondary);
}

.topbar nav a:hover { background: var(--surface-subtle); color: var(--text); text-decoration: none; }
.topbar nav a.active { background: var(--accent-soft); color: var(--accent); }

/* -- stat tiles ----------------------------------------------------------- */

.meta-grid {
  display: flex;
  flex-wrap: wrap;
  gap: .75rem;
  margin: 0 0 1.5rem;
}

.meta-grid > div {
  flex: 1 1 auto;
  min-width: 11.5rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: .65rem .85rem;
}

.meta-grid dt {
  color: var(--muted);
  font-size: .72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .04em;
}

.meta-grid dd {
  margin: .2rem 0 0;
  font-family: ui-monospace, monospace;
  font-size: .88rem;
  overflow-wrap: break-word;
  word-break: break-all;
}


/* -- tables ---------------------------------------------------------------*/

table {
  border-collapse: separate;
  border-spacing: 0;
  width: 100%;
  margin-bottom: 1.25rem;
  font-size: .89rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

th, td {
  text-align: left;
  padding: .55rem .75rem;
  border-bottom: 1px solid var(--border);
}

tbody tr:last-child td { border-bottom: 0; }

th {
  color: var(--muted);
  font-weight: 600;
  font-size: .74rem;
  text-transform: uppercase;
  letter-spacing: .04em;
  background: var(--surface-subtle);
}

td.num, th.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

tbody tr:hover td { background: var(--surface-subtle); }

/* -- badges, bars ----------------------------------------------------------*/

.badge {
  display: inline-flex;
  align-items: center;
  gap: .35em;
  padding: .15rem .55rem;
  border-radius: 999px;
  font-size: .76rem;
  font-weight: 600;
  white-space: nowrap;
}

.badge::before {
  content: "";
  width: .45em;
  height: .45em;
  border-radius: 50%;
  background: currentColor;
}

.badge.good { color: var(--good); background: var(--good-bg); }
.badge.warn { color: var(--warn); background: var(--warn-bg); }
.badge.bad { color: var(--bad); background: var(--bad-bg); }
.badge.neutral { color: var(--muted); background: var(--surface-subtle); }

.bar-track {
  display: inline-block;
  width: 5.5rem;
  height: .4rem;
  border-radius: 999px;
  background: var(--surface-subtle);
  vertical-align: middle;
  overflow: hidden;
}

.bar-fill { display: block; height: 100%; }
.bar-fill.good { background: var(--good); }
.bar-fill.warn { background: var(--warn); }
.bar-fill.bad { background: var(--bad); }

/* -- callouts ---------------------------------------------------------------*/

.callout {
  border: 1px solid var(--border);
  border-left: 3px solid var(--warn);
  background: var(--warn-bg);
  padding: .85rem 1.1rem;
  border-radius: var(--radius-md);
  margin: 1rem 0;
  font-size: .9rem;
}

.callout.bad { border-left-color: var(--bad); background: var(--bad-bg); }
.callout.good { border-left-color: var(--good); background: var(--good-bg); }
.callout ul { margin: .4rem 0 0; padding-left: 1.2rem; }

/* Collapsible callout (Warnings / Not computed): de-emphasised provenance at the
   foot of the report, not an alarm at the top. Neutral, closed by default. */
details.callout.collapsible {
  border-left-color: var(--border);
  background: var(--surface);
  padding: 0;
}

details.callout.collapsible > summary {
  cursor: pointer;
  list-style: none;
  display: flex;
  align-items: center;
  gap: .5rem;
  padding: .7rem 1.1rem;
  color: var(--text-secondary);
  font-size: .9rem;
}

details.callout.collapsible > summary::-webkit-details-marker { display: none; }
details.callout.collapsible > summary::before {
  content: "\\25B8";  /* ▸ */
  color: var(--muted);
  font-size: .8em;
  transition: transform .12s ease;
}
details.callout.collapsible[open] > summary::before { transform: rotate(90deg); }
details.callout.collapsible[open] > summary { border-bottom: 1px solid var(--border); }
details.callout.collapsible > summary:hover { color: var(--text); }

details.callout.collapsible .count {
  margin-left: auto;
  color: var(--muted);
  font-size: .76rem;
  font-weight: 600;
  background: var(--surface-subtle);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: .05rem .5rem;
}

details.callout.collapsible > ul {
  margin: .6rem 0 .7rem;
  padding-left: 2.3rem;
  color: var(--text-secondary);
  font-size: .88rem;
}

.muted { color: var(--muted); }
.small { font-size: .82rem; }
.mono { font-family: ui-monospace, monospace; }

code {
  font-family: ui-monospace, monospace;
  background: var(--surface-subtle);
  border: 1px solid var(--border);
  padding: .1rem .4rem;
  border-radius: 4px;
  font-size: .85em;
}

/* -- run list ---------------------------------------------------------------*/

.run-filter { width: 100%; margin: 0 0 1rem; }

.run-list { list-style: none; padding: 0; display: grid; gap: .6rem; }

.run-list li {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  transition: box-shadow .12s ease, border-color .12s ease;
  display: flex;
  align-items: stretch;
}

.run-list li:hover { box-shadow: var(--shadow-sm); border-color: var(--accent); }

.run-list .run-link {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: .85rem 1.1rem;
  color: var(--text);
  flex: 1 1 auto;
  min-width: 0;
}

/* Per-run delete. Sits outside the run's <a> so it is not a nested control;
   muted until hover, then reads as the destructive action it is. Kept hidden
   from view until the row is hovered/focused to avoid an accidental click. */
.run-del {
  flex: 0 0 auto;
  align-self: center;
  margin: 0 .8rem 0 0;
  padding: .3rem .7rem;
  font-size: .8rem;
  background: transparent;
  color: var(--muted);
  border: 1px solid var(--border);
  opacity: 0;
  transition: opacity .12s ease, color .12s ease, border-color .12s ease, background .12s ease;
}
.run-list li:hover .run-del,
.run-del:focus { opacity: 1; }
.run-del:hover { background: var(--bad-bg); color: var(--bad); border-color: var(--bad); }

.run-list .run-link:hover { text-decoration: none; }
.run-main { min-width: 0; }
.run-list .title { font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.run-list .sub { color: var(--muted); font-size: .82rem; margin-top: .3rem; }

.run-stats {
  display: flex;
  align-items: center;
  gap: .5rem;
  flex-shrink: 0;
}

.run-count { color: var(--muted); font-size: .82rem; font-variant-numeric: tabular-nums; }

.run-empty { color: var(--muted); padding: 1.25rem; text-align: center; }

/* -- theme toggle -------------------------------------------------------- */

.theme-toggle {
  position: fixed;
  bottom: 1.25rem;
  right: 1.25rem;
  z-index: 40;
  padding: .5rem .85rem;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface);
  color: var(--text);
  cursor: pointer;
  font: inherit;
  font-size: .82rem;
  line-height: 1.2;
  white-space: nowrap;
  box-shadow: var(--shadow-md);
  transition: background .12s ease, transform .12s ease;
}

.theme-toggle:hover { background: var(--surface-subtle); transform: translateY(-1px); }
.theme-toggle:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.audio-cell audio { height: 30px; width: 12rem; }

/* -- export bar ---------------------------------------------------------- */

.export-bar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: .5rem;
  margin: 0 0 1.5rem;
}

.export-bar .export-label {
  color: var(--muted);
  font-size: .78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .04em;
  margin-right: .1rem;
}

.export-bar a.btn { text-decoration: none; }
.export-bar a.btn:hover { text-decoration: none; }

/* -- print / save-as-PDF ------------------------------------------------- */

/* @page owns only the vertical margins (they repeat on every page); the left/
   right space is body padding instead, because horizontal @page margins get
   dropped on some print paths — padding is honoured regardless, and splitting
   the two axes avoids the doubled indent you'd get if both applied. */
@page { margin: 1.5cm 0; }

@media print {
  /* Chrome that only makes sense on screen; a printed/PDF'd report is a static
     archival snapshot, so drop the interactive controls and force light ink. */
  html[data-theme="dark"] { color-scheme: light; }
  .topbar nav, .export-bar, .theme-toggle, .run-filter { display: none !important; }

  /* An archival PDF snapshot: sign-off is workflow state, not a benchmark
     result, and the full per-utterance table duplicates the CSV export — both
     just add pages. Warnings are the opposite case: provenance worth reading
     without a click, so force the <details> open regardless of its collapsed
     default on screen. */
  .utterances-section { display: none !important; }

  html, body { background: #fff; color: #000; }
  /* max-width/auto-margins centre on screen; reset them so every block spans the
     same printable width. Horizontal padding here is the actual left/right page
     space (see @page note above); vertical padding stays 0 so @page owns it. */
  body { max-width: none; width: auto; margin: 0; padding: 0 1.5cm; }

  .topbar {
    position: static;
    box-shadow: none;
    display: block;   /* space-between flex leaves a lone h1 pushed right; stack it */
    padding: 0 0 .75rem;
    border: 0;
    border-bottom: 1px solid var(--border);
    border-radius: 0;
  }

  /* Screen uses flex-grow, which adds equal FREE SPACE to each card, not equal
     width — so a short card ("Concurrency: 1") stays narrow and rows end ragged.
     For print, an equal-column grid gives uniform cards with aligned edges that
     span the full printable width. */
  .meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(11.5rem, 1fr));
  }
  .meta-grid > div { min-width: 0; }

  /* Keep a stat card, a table row, or a callout from being split across a page
     boundary — a benchmark table halved between pages is unreadable. */
  .meta-grid > div, tr, .callout, details { break-inside: avoid; }
  table, h2, h3 { break-after: avoid; }
  a { color: inherit; text-decoration: none; }
}

footer {
  margin-top: 3rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: .8rem;
}

/* -- forms + buttons (shared by report compare form and UI pages) ------- */

button, .btn {
  padding: .5rem 1rem;
  border-radius: var(--radius-sm);
  border: 0;
  background: var(--accent);
  color: #fff;
  cursor: pointer;
  font: inherit;
  font-size: .88rem;
  font-weight: 550;
}

button:hover, .btn:hover { background: var(--accent-hover); }
button:disabled, .btn:disabled { opacity: .5; cursor: not-allowed; }
.btn-secondary { background: var(--surface-subtle); color: var(--text); border: 1px solid var(--border); }
.btn-secondary:hover { background: var(--border); }

/* All text-like inputs, not just [type=text]: bare inputs, search and number
   fields were falling through to the default browser chrome before. */
select, textarea, .new-input,
input:not([type="checkbox"]):not([type="radio"]) {
  padding: .45rem .6rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text);
  font: inherit;
  font-size: .9rem;
}

/* Native search "clear" affordance clashes with the dark theme; normalise it. */
input[type="search"] { -webkit-appearance: none; appearance: none; }

select:focus, input:focus, textarea:focus {
  outline: 2px solid var(--accent-soft);
  border-color: var(--accent);
}

.compare-form {
  display: flex;
  gap: .6rem;
  align-items: center;
  flex-wrap: wrap;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1rem 1.1rem;
  margin: 0 0 1.5rem;
}

.form-group { margin-bottom: 1rem; }
.form-group label {
  display: block;
  font-size: .76rem;
  color: var(--muted);
  margin-bottom: .3rem;
  text-transform: uppercase;
  letter-spacing: .04em;
  font-weight: 600;
}
.form-group select, .form-group input { width: 100%; }

/* -- launch progress ------------------------------------------------------*/

.status-box {
  background: var(--surface);
  padding: 1rem 1.1rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  margin: 1.5rem 0;
  font-size: .9rem;
  min-height: 3rem;
}

.progress-bar {
  width: 100%;
  height: 6px;
  background: var(--surface-subtle);
  border-radius: 3px;
  overflow: hidden;
  margin: .6rem 0;
}

.progress-fill { height: 100%; background: var(--accent); transition: width .3s ease; }

/* -- toast ------------------------------------------------------------- */

.toast {
  position: fixed;
  bottom: 4.5rem;
  right: 1.5rem;
  padding: .6rem 1.2rem;
  border-radius: var(--radius-sm);
  font-size: .88rem;
  color: #fff;
  box-shadow: var(--shadow-md);
  z-index: 999;
  animation: tts-eval-fadeout 2.5s forwards;
}

.toast-ok { background: var(--good); }
.toast-err { background: var(--bad); }

@keyframes tts-eval-fadeout {
  0%, 70% { opacity: 1; }
  100% { opacity: 0; }
}

/* -- config editor -------------------------------------------------------*/

.config-grid { display: grid; grid-template-columns: 15.5rem 1fr; gap: 1.25rem; margin: 1rem 0; }
.config-section { margin-bottom: 1.5rem; }
.config-section h3 { margin: 0 0 .5rem; }

.config-list { list-style: none; padding: 0; margin: 0; display: grid; gap: .35rem; }
.config-list li {
  padding: .5rem .7rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: .87rem;
  font-family: ui-monospace, monospace;
}
.config-list li:hover { border-color: var(--accent); }
.config-list li.active { border-color: var(--accent); background: var(--accent-soft); color: var(--accent); }

.editor-wrap { position: relative; min-height: 450px; }

.editor-area, .editor-highlight {
  width: 100%;
  height: 100%;
  min-height: 450px;
  margin: 0;
  font-family: ui-monospace, monospace;
  font-size: .87rem;
  line-height: 1.55;
  padding: .85rem;
  border: 1px solid transparent;
  border-radius: var(--radius-md);
  white-space: pre-wrap;
  word-break: break-word;
  tab-size: 2;
}

.editor-area {
  position: relative;
  background: transparent;
  color: transparent;
  caret-color: var(--text);
  border-color: var(--border);
  resize: vertical;
}

.editor-highlight {
  position: absolute;
  inset: 0;
  overflow: hidden;
  background: var(--surface);
  color: var(--text);
  pointer-events: none;
}

.editor-area:disabled ~ .editor-highlight { color: var(--muted); }

.yaml-key { color: var(--syntax-key); font-weight: 600; }
.yaml-string { color: var(--syntax-string); }
.yaml-number, .yaml-bool { color: var(--syntax-number); }
.yaml-comment { color: var(--syntax-comment); font-style: italic; }
.yaml-env { color: var(--syntax-env); }
.yaml-dash { color: var(--syntax-dash); font-weight: 600; }

.editor-toolbar { display: flex; gap: .5rem; align-items: center; margin-bottom: .75rem; }
.editor-title { font-weight: 600; font-size: .95rem; flex: 1; }
.new-input { width: 100%; margin-bottom: .4rem; }

/* -- new-config dialog ---------------------------------------------------- */

dialog.modal {
  width: min(40rem, calc(100vw - 2rem));
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  background: var(--surface);
  color: var(--text);
  box-shadow: var(--shadow-md);
  padding: 0;
}

dialog.modal::backdrop { background: rgba(10, 12, 16, .5); backdrop-filter: blur(2px); }

.modal-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  padding: 1.1rem 1.25rem .5rem;
}

.modal-head h3 { margin: 0; font-size: 1.05rem; }
.modal-head p { margin: .15rem 0 0; }

.modal-body { padding: .5rem 1.25rem 0; }
.modal-body .form-group { margin-bottom: .85rem; }

.yaml-input {
  width: 100%;
  min-height: 17rem;
  font-family: ui-monospace, monospace;
  font-size: .85rem;
  line-height: 1.55;
  tab-size: 2;
  white-space: pre;
  resize: vertical;
}

.modal-foot {
  display: flex;
  justify-content: flex-end;
  gap: .6rem;
  padding: 1rem 1.25rem 1.25rem;
}

.modal-hint { font-size: .82rem; color: var(--muted); margin: 0 0 .6rem; }
"""

__all__ = ["CSS"]
