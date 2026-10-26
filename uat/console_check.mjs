#!/usr/bin/env node
// Dashboard console-error check — issue #293, increment 1.
//
// Drives a browser at an ALREADY RUNNING controller (CI boots one from
// uat/Dockerfile; see .github/workflows/ci.yml) and fails on any browser console
// error, uncaught exception or failed same-origin request while walking every
// tab. It boots nothing itself.
//
// Auth is the DOCUMENTED first-run path, not a bypass: the controller prints a
// one-time bootstrap token to stdout on a database with no users, POST /api/setup
// exchanges it for an admin session, and the session goes into localStorage under
// `em_token` exactly as the landing page stores it. The device is created the
// documented way too — /api/provision/tls_credentials is what the provisioning
// wizard calls to mint a pending row from a serial — then approved. A fake
// device is a script rather than hardware; the WebSocket protocol is ours.
//
//   EM_UAT_URL=http://em-uat:8768 EM_UAT_BOOTSTRAP=<64 hex> node console_check.mjs
//
// Env: EM_UAT_URL (default http://127.0.0.1:8768), EM_UAT_BOOTSTRAP (from the
// controller log; only needed while the database has no users), EM_UAT_DEVICE
// (serial), EM_UAT_LABEL (device label).

import { chromium } from 'playwright';

const BASE   = (process.env.EM_UAT_URL || 'http://127.0.0.1:8768').replace(/\/$/, '');
const SERIAL = process.env.EM_UAT_DEVICE || 'em-uat-01';
const LABEL  = process.env.EM_UAT_LABEL  || 'UAT';

// Fixed because this runs against a throwaway database in CI: the controller
// exists for the length of the job, so there is no password to protect and
// nothing to rotate.
const USER = 'uat';
const PASS = 'uat-console-check';

// The two tab sets, read out of dashboard.jsx (Detail's TABS and Settings'
// TAB_LABELS — the first is the raw key, the second is a label; TabBar is the
// class both strips render through).
const DEVICE_TABS   = ['status', 'activity', 'config', 'console', 'updates', 'logs'];
const SETTINGS_TABS = ['Config', 'System', 'Users', 'Account', 'Support'];

const problems = [];
const visited = new Set();

const body = r => r.body.replace(/\s+/g, ' ').slice(0, 200);

async function api(path, opts = {}) {
  const res = await fetch(BASE + path, {
    ...opts,
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
  });
  const body = await res.text();
  let json = null;
  try { json = JSON.parse(body); } catch (e) { /* an HTML error page */ }
  return { status: res.status, json, body };
}

async function session() {
  const bootstrap = process.env.EM_UAT_BOOTSTRAP;
  if (bootstrap) {
    const r = await api('/api/setup', {
      method: 'POST',
      body: JSON.stringify({ token: bootstrap, username: USER, password: PASS }),
    });
    if (r.status === 201) return r.json.token;
    // 403/409 means setup is already done — a controller somebody built before,
    // which is the normal case when this is run by hand rather than in CI.
    if (r.status !== 403 && r.status !== 409) {
      throw new Error(`/api/setup returned ${r.status}: ${body(r)}`);
    }
    console.log(`setup already done (${r.status}) — logging in instead`);
  }
  const r = await api('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username: USER, password: PASS }),
  });
  if (r.status !== 200) throw new Error(`/api/auth/login returned ${r.status}: ${body(r)}`);
  return r.json.token;
}

async function seedDevice(token) {
  const h = { Authorization: `Bearer ${token}` };
  const creds = await api('/api/provision/tls_credentials', {
    method: 'POST', headers: h, body: JSON.stringify({ device_id: SERIAL }),
  });
  // 503 is a controller with no PKI, which would leave no device row and so no
  // device tabs at all.
  if (creds.status !== 200) {
    throw new Error(`tls_credentials returned ${creds.status}: ${body(creds)}`);
  }
  const approved = await api(`/api/devices/${SERIAL}/approve`, {
    method: 'POST', headers: h, body: JSON.stringify({ label: LABEL }),
  });
  if (approved.status !== 200 && approved.status !== 409) { // 409 = already approved
    throw new Error(`approve returned ${approved.status}: ${body(approved)}`);
  }
}

const token = await session();
await seedDevice(token);

const browser = await chromium.launch();
// Printed because "which browser" is the first question when a red build turns
// out to be the browser rather than the dashboard.
console.log(`chromium ${browser.version()}`);

const ctx = await browser.newContext({
  // Fixed viewport, because a console error that only appears at one width is
  // still a console error and the next increment's screenshot baselines need
  // this number written down.
  viewport: { width: 1280, height: 900 },
  reducedMotion: 'reduce',
});
// The dashboard animates unconditionally — inline transitions and CSS keyframes —
// so prefers-reduced-motion alone does not stop it, and a control observed
// mid-fade is a control nobody can reproduce.
await ctx.addInitScript(() => {
  addEventListener('DOMContentLoaded', () => {
    const s = document.createElement('style');
    s.textContent = '*,*::before,*::after{transition:none!important;animation:none!important}';
    document.head.appendChild(s);
  });
});
// The keys the landing page writes on sign-in, so the app boots logged in
// rather than showing the login form. `em_auth_via` is one of them: index.html's
// enterDashboard() stores all three, and the dashboard keys its Sign out control
// on it.
await ctx.addInitScript(
  ([t]) => {
    localStorage.setItem('em_token', t);
    localStorage.setItem('em_role', 'admin');
    localStorage.setItem('em_auth_via', 'password');
  },
  [token],
);

const page = await ctx.newPage();

// WHAT FAILS. Two sources, not one: `console` sees what the page logs and
// `pageerror` sees an uncaught exception, which never goes through console —
// a check with only the first misses every TypeError, and a TypeError is most
// of what this is for.
//
// WHAT IS FILTERED: nothing, deliberately, and the list below is where a real
// exception goes. Measured on this tree over a full walk of all eleven tabs:
// zero console errors, zero page exceptions, and every same-origin response
// 2xx — including static/favicon.svg, which is vendored rather than fetched
// from a CDN. So there is no benign case to carve out today, and inventing one
// would hide the next regression behind a rule written for a fault that does
// not exist. Add an entry with the reason beside it when one shows up.
//
// THE LIMIT, stated because the issue assumes otherwise: this catches a JS
// exception and a failed request. It does NOT catch an invalid CSS VALUE.
// `${color}88` against a `var()` token is dropped by Chromium at computed-value
// time with nothing on the console at all — verified, both as an inline
// declaration and as a custom property — so the LCD glow this was asked to
// catch cannot be caught here. The two checks that can catch it are the source
// lint in tests/test_design_tokens.py and the screenshot baseline in increment
// 2, which sees the missing pixels.
page.on('console', m => {
  if (m.type() === 'error') problems.push(`console: ${m.text()}`);
});
page.on('pageerror', e => problems.push(`pageerror: ${e.message}`));
// A failed request also logs its own console error, but the text is only
// "Failed to load resource" — the URL is what makes a red build diagnosable.
// Same-origin only: everything the dashboard loads is vendored into the image,
// and a cross-origin failure here is a fact about the runner's network rather
// than about the dashboard.
page.on('response', r => {
  if (r.status() >= 400 && r.url().startsWith(BASE)) {
    problems.push(`http ${r.status()}: ${r.url().slice(BASE.length)}`);
  }
});
page.on('requestfailed', r => {
  if (r.url().startsWith(BASE)) problems.push(`request failed: ${r.url().slice(BASE.length)}`);
});

// A tab is reached, not waited for: the dashboard fetches on tab entry, so the
// button being clickable is the observable. 10s is a ceiling rather than a wait,
// because a tab that renders in 80ms should not cost 750ms of wall clock.
//
// A tab that is not there is recorded and the walk carries on, so a renamed tab
// reports as one line per missing tab rather than a Playwright stack trace:
// this check reporting green having visited three of eleven tabs is worse than
// no check, which is the same rule every off-target tool in emos/init carries.
async function openTab(group, label) {
  const tab = page.locator('.em-tabs button', { hasText: new RegExp(`^${label}$`, 'i') }).first();
  try {
    await tab.waitFor({ state: 'visible', timeout: 10_000 });
  } catch (e) {
    problems.push(`never reached: ${group}/${label.toLowerCase()}`);
    return;
  }
  await tab.click();
  await page.waitForTimeout(750);
  visited.add(`${group}/${label.toLowerCase()}`);
}

await page.goto(BASE, { waitUntil: 'domcontentloaded' });
// The tab bars live inside modals, so the fleet grid has to render first, and a
// wait that cannot succeed must fail loudly rather than time out into a report
// of "no console errors" over a page that never loaded.
const tile = page.getByText(LABEL, { exact: true }).first();
await tile.waitFor({ state: 'visible', timeout: 20_000 })
  .catch(() => { throw new Error(`device tile "${LABEL}" never rendered`); });

await tile.click();
await page.waitForSelector('.em-tabs', { timeout: 20_000 });
for (const t of DEVICE_TABS) await openTab('device', t);

// Close the detail modal before opening Settings: both tab bars are in the DOM
// at once and openTab takes the first match, so the device bar would swallow
// every Settings click.
await page.locator('.em-modal button[title="Close"]').first().click();
await page.locator('.em-modal').waitFor({ state: 'detached', timeout: 10_000 });

await page.getByRole('button', { name: 'Settings' }).click();
await page.waitForSelector('.em-tabs', { timeout: 20_000 });
for (const t of SETTINGS_TABS) await openTab('settings', t);

await browser.close();

if (problems.length) {
  console.log(`\n${problems.length} console problem(s):`);
  for (const p of problems) console.log(`  ${p}`);
  process.exit(1);
}
console.log(`clean: ${visited.size} tabs, no console errors`);
