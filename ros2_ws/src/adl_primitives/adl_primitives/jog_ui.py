"""Browser joystick UI for Cartesian jogging (Flask, default port 8080).

Drag the round pad to move the hand in the robot's base-frame XY, drag the
side strip for up/down; an explicit ENABLE toggle arms the joystick (the
server boots disarmed). Velocity streams while you hold; releasing (or losing
the connection) stops the arm — the node's watchdog zeroes the twist if the
stream pauses for ``twist_timeout_s``. Twist commands carry a control lease
token and a sequence number, so a second tab cannot silently keep the arm
moving and a delayed packet can never override the release-zero.

Two backends (``twist_backend`` param / ``sim:=true`` launch arg):

* ``kortex`` (default): ros2_kortex's ``twist_controller``, switched in/out
  with STRICT semantics and post-switch verification; commands are rotated
  into the Kinova TOOL frame via TF (the driver hardcodes that frame) and the
  node streams continuously because the base latches its last twist. REAL
  hardware only.
* ``sim_jtc``: differential IK (damped least squares on a TF/URDF Jacobian)
  streamed as small position steps through the trajectory controller — for
  fake-hardware testing, e.g. watching the arm move in Foxglove.

The per-joint /api/jog endpoint still exists for scripts; the UI no longer
exposes it.

Every command goes through :class:`KinovaPrimitives` — speed/step clamps,
``dry_run`` gate (defaults TRUE: nothing moves), soft-stop.

SAFETY: the web soft-stop is a convenience, NOT a substitute for the hardware
E-stop. If this PROCESS is killed uncleanly (SIGKILL/OOM/power) while the
joystick is held, nothing in software stops the arm — the hardware E-stop is
the only backstop. There is no authentication — anyone who can reach the port
can move the arm. Keep it on the robot LAN (or set ``ui_host`` to 127.0.0.1).
"""

import math
import secrets
import signal
import threading
import time
from urllib.parse import urlsplit

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions

from flask import Flask, jsonify, request

from adl_primitives.kinova_primitives import KinovaPrimitives

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAMMP jog</title>
<style>
  :root { --bg: #0b0d12; --deck: #161a22; --deck2: #1d2330; --edge: #2a3140;
          --ink: #e8ecf3; --dim: #8b94a7; --accent: #38bdf8; --ok: #34d399;
          --warn: #fbbf24; --danger: #ef4444; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; color: var(--ink);
         margin: 0; padding: 14px; min-height: 100vh;
         display: flex; justify-content: center; align-items: flex-start;
         background: radial-gradient(1100px 700px at 50% -10%, #182030 0%, var(--bg) 60%); }
  #deck { width: 100%; max-width: 780px; border-radius: 28px; padding: 18px 20px 20px;
          background: linear-gradient(180deg, var(--deck2), var(--deck));
          border: 1px solid var(--edge);
          box-shadow: 0 24px 60px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.05); }
  header { display: flex; align-items: center; justify-content: space-between;
           margin-bottom: 10px; }
  header h1 { font-size: .9rem; letter-spacing: .3em; text-transform: uppercase;
              color: var(--dim); margin: 0; font-weight: 600; }
  .leds { display: flex; gap: 14px; font-size: .68rem; color: var(--dim);
          text-transform: uppercase; letter-spacing: .08em; align-items: center; }
  .led { display: flex; align-items: center; gap: 6px; }
  .led i { width: 9px; height: 9px; border-radius: 50%; background: #3a4152;
           box-shadow: 0 0 6px transparent; }
  .led.on i { background: var(--ok); box-shadow: 0 0 8px var(--ok); }
  .led.arm.on i { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
  .led.bad i { background: var(--danger); box-shadow: 0 0 8px var(--danger); }
  #banner { padding: 9px 14px; border-radius: 12px; margin-bottom: 12px;
            display: none; font-weight: 600; font-size: .85rem; }
  #banner.dry { display: block; background: #4a3a08; color: var(--warn); }
  #banner.stopped { display: block; background: #4a1414; color: #fca5a5; }
  #banner.sick { display: block; background: #4a1414; color: #fca5a5; }
  button { touch-action: manipulation; user-select: none; -webkit-user-select: none;
           font: inherit; }
  .board { display: flex; gap: 16px; align-items: center;
           justify-content: space-between; flex-wrap: wrap; margin: 8px 0 4px; }
  .stick-col { display: flex; flex-direction: column; align-items: center; gap: 8px;
               flex: 1; min-width: 232px; }
  #panel-joy.dim .stick-col { opacity: .35; }
  #panel-joy.dim #pad, #panel-joy.dim #zstrip { pointer-events: none; }
  .caption { font-size: .68rem; letter-spacing: .22em; text-transform: uppercase;
             color: var(--dim); }
  #pad { width: 230px; height: 230px; border-radius: 50%; position: relative;
         touch-action: none; border: 1px solid var(--edge);
         background: radial-gradient(circle at 50% 40%, #232a37 0%, #12151c 68%, #0d1016 100%);
         box-shadow: inset 0 14px 28px rgba(0,0,0,.6),
                     inset 0 -6px 14px rgba(255,255,255,.03), 0 8px 18px rgba(0,0,0,.4); }
  #pad::after { content: ""; position: absolute; inset: 19%; border-radius: 50%;
                border: 1px dashed #2b3342; pointer-events: none; }
  .lbl { position: absolute; color: var(--dim); font-size: .66rem;
         letter-spacing: .12em; text-transform: uppercase; pointer-events: none;
         user-select: none; -webkit-user-select: none; }
  #pad .n { top: 9px; left: 50%; transform: translateX(-50%); }
  #pad .s { bottom: 9px; left: 50%; transform: translateX(-50%); }
  #pad .w { left: 10px; top: 50%; transform: translateY(-50%); }
  #pad .e { right: 10px; top: 50%; transform: translateY(-50%); }
  .knob { border-radius: 50%; position: absolute; left: 50%; top: 50%;
          transform: translate(-50%, -50%); pointer-events: none;
          background: radial-gradient(circle at 35% 30%, #4b5567, #242a37 60%, #181d27);
          border: 1px solid #3a4452;
          box-shadow: 0 10px 18px rgba(0,0,0,.55), inset 0 3px 6px rgba(255,255,255,.09); }
  .knob::after { content: ""; position: absolute; inset: 34%; border-radius: 50%;
                 background: #10141b; box-shadow: inset 0 2px 5px rgba(0,0,0,.7); }
  #dot { width: 86px; height: 86px; }
  #zdot { width: 68px; height: 68px; }
  .knob.spring { transition: left .18s ease-out, top .18s ease-out; }
  #zstrip { width: 86px; height: 230px; border-radius: 999px; position: relative;
            touch-action: none; border: 1px solid var(--edge);
            background: linear-gradient(180deg, #12151c, #1c222e 50%, #12151c);
            box-shadow: inset 0 12px 24px rgba(0,0,0,.55), 0 8px 18px rgba(0,0,0,.4); }
  #zstrip .u { top: 10px; left: 50%; transform: translateX(-50%); }
  #zstrip .d { bottom: 10px; left: 50%; transform: translateX(-50%); }
  .cluster { display: flex; flex-direction: column; align-items: center; gap: 10px;
             min-width: 180px; }
  #stop { width: 118px; height: 118px; border-radius: 50%; border: none;
          cursor: pointer; color: white; font-weight: 800; font-size: .92rem;
          letter-spacing: .05em; line-height: 1.2; padding: 8px;
          background: radial-gradient(circle at 35% 28%, #ff8a7a, var(--danger) 55%, #b91c1c);
          box-shadow: 0 10px 26px rgba(239,68,68,.35),
                      inset 0 -6px 12px rgba(0,0,0,.35), inset 0 2px 4px rgba(255,255,255,.35); }
  #stop:active { transform: translateY(2px); }
  .pill { width: 170px; padding: 10px 12px; border-radius: 999px; cursor: pointer;
          border: 1px solid var(--edge); background: var(--deck2); color: var(--ink);
          font-size: .82rem; font-weight: 600; }
  .pill:active { transform: translateY(1px); }
  #enable.on { background: #0b4a6f; border-color: var(--accent); color: #bae6fd;
               box-shadow: 0 0 14px rgba(56,189,248,.25); }
  #resume { border-color: #14532d; color: #86efac; }
  .meter { display: flex; align-items: center; gap: 12px; margin: 14px 2px 10px; }
  .meter input { flex: 1; accent-color: var(--accent); height: 4px; }
  #speedlbl { font-variant-numeric: tabular-nums; color: var(--accent);
              font-weight: 700; min-width: 74px; text-align: right; }
  .actions { display: flex; gap: 10px; }
  button.grip { flex: 1; padding: 13px; font-size: .92rem; font-weight: 600;
                border: 1px solid var(--edge); border-radius: 14px;
                background: linear-gradient(180deg, #222937, #1a202b);
                color: var(--ink); cursor: pointer;
                box-shadow: 0 4px 10px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.05); }
  button.grip:active { transform: translateY(1px); }
  button.grip:disabled { opacity: .35; cursor: default; }
  #msg { min-height: 1.3em; color: var(--warn); margin-top: 10px; font-weight: 700;
         text-align: center; font-size: .88rem; }
  .hint { color: var(--dim); font-size: .72rem; text-align: center; margin: 8px 0 0; }
  @media (max-width: 640px) {
    .cluster { order: -1; width: 100%; }
    .board { justify-content: center; }
  }
</style>
</head>
<body>
<div id="deck">
<header>
  <h1>RAMMP&nbsp;Controller</h1>
  <div class="leds">
    <span class="led" id="led-link"><i></i>link</span>
    <span class="led arm" id="led-arm"><i></i>armed</span>
    <span class="led" id="led-sim" style="display:none"><i></i>sim</span>
  </div>
</header>
<div id="banner"></div>

<div class="board dim" id="panel-joy">
  <div class="stick-col">
    <div id="pad">
      <span class="lbl n">forward</span><span class="lbl s">back</span>
      <span class="lbl w">left</span><span class="lbl e">right</span>
      <div id="dot" class="knob"></div>
    </div>
    <span class="caption">move</span>
  </div>

  <div class="cluster">
    <button id="stop">SOFT-STOP</button>
    <button id="enable" class="pill">ENABLE JOYSTICK</button>
    <button id="home" class="pill">Home pose</button>
    <button id="resume" class="pill">Resume controller</button>
  </div>

  <div class="stick-col">
    <div id="zstrip">
      <span class="lbl u">up</span><span class="lbl d">down</span>
      <div id="zdot" class="knob"></div>
    </div>
    <span class="caption">height</span>
  </div>
</div>

<div class="meter">
  <span class="caption">speed</span>
  <input type="range" id="speed" min="10" max="100" step="10" value="30">
  <span id="speedlbl"></span>
</div>

<div class="actions">
  <button class="grip" id="gopen">&#10096;&#8201;&#10097;&nbsp; Gripper open</button>
  <button class="grip" id="gclose">&#10097;&#8201;&#10096;&nbsp; Gripper close</button>
</div>
<p class="hint" id="bhint"></p>
<p class="hint">Hold to move &middot; release to stop &middot; directions are the robot's base frame</p>
<div id="msg"></div>
</div>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
let built = false, inflight = 0, stopping = false, lostConn = false, lastStatus = null;
let mode = "joint", maxLin = 0.05;
let twistToken = null, twistSeq = 0, twistInFlight = false;

async function api(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    const err = new Error(j.error || ("HTTP " + r.status));
    err.status = r.status;
    throw err;
  }
  return j;
}

function setMsg(text) { $("msg").textContent = text || ""; }

function setControlsDisabled(d) {
  document.querySelectorAll("button.grip, #home").forEach((b) => (b.disabled = d));
}

/* ------------------------------------------------ joystick (cartesian) */
let padVec = {x: 0, y: 0}, zVal = 0, padPointer = null, zPointer = null, streamer = null;

function speedMps() { return maxLin * ($("speed").value / 100); }
function updateSpeedLbl() {
  $("speedlbl").textContent = (speedMps() * 100).toFixed(1) + " cm/s";
}

function drawDot() {
  // Visual travel is capped short of the rim so the knob stays in its well;
  // the commanded padVec/zVal still reach the full -1..1 range.
  $("dot").style.left = (padVec.x * 31 + 50) + "%";
  $("dot").style.top = (padVec.y * 31 + 50) + "%";
  $("zdot").style.top = (-zVal * 33 + 50) + "%";
}

function twistBody() {
  const sp = speedMps();
  // Screen-to-base-frame: pad up = +X (forward), pad right = -Y (robot right),
  // strip up = +Z. (The server rotates base-frame into the tool frame via TF.)
  return {
    token: twistToken,
    seq: ++twistSeq,
    vx: -padVec.y * sp,
    vy: -padVec.x * sp,
    vz: zVal * sp,
  };
}

async function sendTwist() {
  if (twistInFlight) return;  // never stack requests: ordering is seq-enforced anyway
  twistInFlight = true;
  try { await api("/api/twist", twistBody()); }
  catch (e) { setMsg(e.message); }
  finally { twistInFlight = false; }
}

async function zeroConfirm() {
  // The release-zero is the primary stop affordance: await it, retry it.
  for (let i = 0; i < 3; i++) {
    try {
      await api("/api/twist", {token: twistToken, seq: ++twistSeq, vx: 0, vy: 0, vz: 0});
      return;
    } catch (e) {
      // 409 = soft-stopped / mode already off / lease superseded: in every one
      // of those server states our stream is already dead — that IS a stop.
      if (e.status === 409) return;
      await new Promise((r) => setTimeout(r, 150));
    }
  }
  setMsg("release-zero not confirmed - watchdog will stop the arm");
}

function startStream() {
  if (!streamer) {
    sendTwist();
    streamer = setInterval(sendTwist, 100);
  }
}

function releaseAll() {
  padPointer = null; zPointer = null;
  padVec = {x: 0, y: 0}; zVal = 0;
  $("dot").classList.add("spring");
  $("zdot").classList.add("spring");
  drawDot();
  if (streamer) { clearInterval(streamer); streamer = null; }
  if (twistToken) zeroConfirm();
}

function bindPad() {
  const pad = $("pad"), zstrip = $("zstrip");
  [pad, zstrip].forEach((el) => el.addEventListener("contextmenu", (e) => e.preventDefault()));

  const padMove = (e) => {
    const r = pad.getBoundingClientRect();
    let x = ((e.clientX - r.left) / r.width) * 2 - 1;
    let y = ((e.clientY - r.top) / r.height) * 2 - 1;
    const m = Math.hypot(x, y);
    if (m > 1) { x /= m; y /= m; }
    padVec = {x: x, y: y};
    drawDot();
  };
  pad.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || padPointer !== null) return;  // one owning pointer only
    padPointer = e.pointerId;
    pad.setPointerCapture(e.pointerId);
    $("dot").classList.remove("spring");
    padMove(e); startStream();
  });
  pad.addEventListener("pointermove", (e) => {
    if (e.pointerId === padPointer) padMove(e);
  });
  ["pointerup", "pointercancel"].forEach((ev) => pad.addEventListener(ev, (e) => {
    if (e.pointerId !== padPointer) return;
    padPointer = null; padVec = {x: 0, y: 0};
    $("dot").classList.add("spring");
    drawDot();
    if (zPointer === null) releaseAll(); else sendTwist();
  }));

  const zMove = (e) => {
    const r = zstrip.getBoundingClientRect();
    let y = ((e.clientY - r.top) / r.height) * 2 - 1;
    zVal = Math.max(-1, Math.min(1, -y));
    drawDot();
  };
  zstrip.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || zPointer !== null) return;
    zPointer = e.pointerId;
    zstrip.setPointerCapture(e.pointerId);
    $("zdot").classList.remove("spring");
    zMove(e); startStream();
  });
  zstrip.addEventListener("pointermove", (e) => {
    if (e.pointerId === zPointer) zMove(e);
  });
  ["pointerup", "pointercancel"].forEach((ev) => zstrip.addEventListener(ev, (e) => {
    if (e.pointerId !== zPointer) return;
    zPointer = null; zVal = 0;
    $("zdot").classList.add("spring");
    drawDot();
    if (padPointer === null) releaseAll(); else sendTwist();
  }));

  // Any way the page loses the user's attention: stop the arm.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) releaseAll();
  });
  window.addEventListener("pagehide", releaseAll);
  window.addEventListener("blur", releaseAll);
  $("speed").addEventListener("input", updateSpeedLbl);
  updateSpeedLbl();
}

/* ------------------------------------------------ gripper */
async function grip(action) {
  if (inflight) return;
  inflight++;
  setControlsDisabled(true);
  try {
    setMsg("gripper " + action + " ...");
    await api("/api/gripper", {action: action});
    setMsg("");
  } catch (e) { setMsg(e.message); }
  finally { inflight--; }
}

/* ------------------------------------------------ mode + stop + status */
async function setMode(m) {
  releaseAll();
  try {
    const j = await api("/api/mode", {mode: m});
    twistToken = j.token || null;
    twistSeq = 0;
    setMsg("");
  } catch (e) { setMsg(e.message); }
}

async function stopArm() {
  // Retry hard: a soft-stop lost to a WiFi hiccup must not fail silently.
  if (stopping) return;
  stopping = true;
  releaseAll();
  $("stop").textContent = "STOPPING...";
  for (let i = 0; i < 10; i++) {
    try {
      await api("/api/estop");
      stopping = false;
      setMsg("");
      return;
    } catch (e) {
      setMsg("stop not confirmed - retrying (" + (i + 1) + "/10)");
      await new Promise((r) => setTimeout(r, 250));
    }
  }
  stopping = false;
  setMsg("STOP NOT CONFIRMED - USE THE HARDWARE E-STOP");
  if (lastStatus && !lastStatus.stopped) $("stop").textContent = "SOFT-STOP";
}

function render(s) {
  lastStatus = s;
  maxLin = s.max_linear_mps;
  if (!built) { bindPad(); built = true; }
  mode = s.mode;
  if (mode !== "cartesian" && twistToken) {
    // The server disarmed us (estop, takeover, restart): drop the stale lease
    // quietly so blur/tab-switch doesn't fire a doomed zeroConfirm later.
    twistToken = null;
    if (streamer) { clearInterval(streamer); streamer = null; }
  }
  const on = mode === "cartesian";
  $("panel-joy").className = on ? "board" : "board dim";
  $("enable").textContent = on ? "JOYSTICK ENABLED — tap to disable" : "ENABLE JOYSTICK";
  $("enable").className = on ? "pill on" : "pill";
  $("led-arm").className = on ? "led arm on" : "led arm";
  const sim = s.backend === "sim_jtc";
  $("led-sim").style.display = sim ? "" : "none";
  if (sim) $("led-sim").className = "led on";
  $("bhint").textContent = sim
    ? "SIM backend: differential IK through the trajectory controller (fake hardware)"
    : "";
  const banner = $("banner");
  if (s.healthy === false) {
    banner.className = "sick";
    banner.textContent =
      "JOG NODE UNHEALTHY — twist disabled and soft-stopped. Restart jog_ui. " +
      "If the arm is moving, use the HARDWARE E-STOP.";
  } else if (s.stopped) {
    banner.className = "stopped";
    banner.textContent =
      "SOFT-STOPPED — goals cancelled; controller deactivation requested. Resume to continue.";
  } else if (s.dry_run) {
    banner.className = "dry";
    banner.textContent =
      "DRY RUN — commands are logged, nothing moves (dry_run:=false for motion)";
  } else {
    banner.className = "";
  }
  $("stop").textContent =
    s.stopped ? "STOPPED (soft)" : (stopping ? "STOPPING..." : "SOFT-STOP");
  setControlsDisabled(s.stopped || s.busy || inflight > 0 || !s.have_joint_state);
}

$("stop").addEventListener("click", stopArm);
$("resume").addEventListener("click", async () => {
  try { await api("/api/resume", {confirm: true}); setMsg(""); }
  catch (e) { setMsg(e.message); }
});
$("gopen").addEventListener("click", () => grip("open"));
$("gclose").addEventListener("click", () => grip("close"));
$("home").addEventListener("click", async () => {
  if (inflight) return;
  inflight++;
  setControlsDisabled(true);
  try {
    setMsg("moving to home pose ...");
    await api("/api/home", {confirm: true});
    setMsg("");
  } catch (e) { setMsg(e.message); }
  finally { inflight--; }
});
$("enable").addEventListener("click", () =>
  setMode(mode === "cartesian" ? "joint" : "cartesian"));

async function poll() {
  try {
    const s = await (await fetch("/api/status")).json();
    if (lostConn) { lostConn = false; setMsg(""); }
    $("led-link").className = "led on";
    render(s);
  } catch (e) {
    lostConn = true;
    $("led-link").className = "led bad";
    setMsg("connection lost");
  }
  setTimeout(poll, 500);
}
poll();
</script>
</body>
</html>
"""

TICK_STALE_S = 0.5  # executor considered dead if the 20 Hz tick is older than this


def main(args=None):
    # Handle SIGINT ourselves so soft_stop() still has a live context (see test_arm).
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = KinovaPrimitives(node_name='jog_ui')
    ui_host = node.declare_parameter('ui_host', '0.0.0.0').value
    ui_port = node.declare_parameter('ui_port', 8080).value
    max_step_deg = node.declare_parameter('max_step_deg', 5.0).value
    step_time_s = node.declare_parameter('step_time_s', 2.0).value

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def _spin():
        # The 20 Hz twist streamer and its watchdog live on this thread: if it
        # dies, the deadman chain is gone — stop the arm before going quiet.
        try:
            executor.spin()
        except Exception as exc:
            node.get_logger().fatal('Executor spin died: %s — soft-stopping.' % exc)
            try:
                node.soft_stop()
            except Exception:
                pass

    spin_thread = threading.Thread(target=_spin, daemon=True)
    spin_thread.start()

    busy = threading.Lock()  # one blocking motion command at a time
    lease_lock = threading.Lock()
    lease = {'token': None, 'seq': 0}  # control lease for /api/twist
    app = Flask(__name__)

    def origin_ok():
        """Reject cross-site browser requests (CSRF/DNS-rebinding guard).

        Same-origin fetches carry an Origin matching the Host; non-browser
        clients typically send no Origin at all and are allowed (this is a
        CSRF guard, not authentication).
        """
        origin = request.headers.get('Origin')
        if not origin:
            return True
        return urlsplit(origin).netloc == request.host

    def node_healthy():
        return spin_thread.is_alive() and node.tick_age() < TICK_STALE_S

    @app.get('/')
    def index():
        return PAGE

    @app.get('/api/status')
    def status():
        positions = node.get_current_positions()
        return jsonify({
            'joint_names': list(node.joint_names),
            'positions_deg': (
                [math.degrees(p) for p in positions] if positions is not None else None
            ),
            'have_joint_state': positions is not None,
            'dry_run': node.dry_run,
            'stopped': node.stop_requested(),
            'busy': busy.locked(),
            'mode': 'cartesian' if node.cartesian_active() else 'joint',
            'healthy': node_healthy(),
            'backend': node.twist_backend,
            'max_step_deg': max_step_deg,
            'max_linear_mps': node.max_linear_mps,
        })

    @app.post('/api/mode')
    def mode():
        if not origin_ok():
            return jsonify(ok=False, error='cross-origin request rejected'), 403
        data = request.get_json(silent=True)
        if data is None or data.get('mode') not in ('joint', 'cartesian'):
            return jsonify(ok=False, error="mode must be 'joint' or 'cartesian'"), 400
        if node.stop_requested():
            return jsonify(ok=False, error='soft-stopped; resume first'), 409
        if not busy.acquire(blocking=False):
            return jsonify(ok=False, error='a move is already in progress'), 409
        try:
            want = data['mode']
            if want == 'cartesian':
                if not node.cartesian_active() and not node.activate_cartesian():
                    return jsonify(ok=False, error='controller switch failed', mode='joint'), 500
                # Issue (or reissue) the control lease. A same-mode request is a
                # TAKEOVER: the previous client's token stops working and its
                # stream dies on the watchdog.
                with lease_lock:
                    lease['token'] = secrets.token_hex(8)
                    lease['seq'] = 0
                return jsonify(ok=True, mode='cartesian', token=lease['token'])
            # want == 'joint'
            if node.cartesian_active() and not node.deactivate_cartesian():
                return jsonify(ok=False, error='controller switch failed', mode='cartesian'), 500
            with lease_lock:
                lease['token'] = None
                lease['seq'] = 0
            return jsonify(ok=True, mode='joint')
        finally:
            busy.release()

    @app.post('/api/twist')
    def twist():
        if not origin_ok():
            return jsonify(ok=False, error='cross-origin request rejected'), 403
        data = request.get_json(silent=True)
        if data is None:
            return jsonify(ok=False, error='expected an application/json body'), 400
        try:
            vx = float(data.get('vx', 0.0))
            vy = float(data.get('vy', 0.0))
            vz = float(data.get('vz', 0.0))
            seq = int(data.get('seq', -1))
        except (TypeError, ValueError):
            return jsonify(ok=False, error='vx/vy/vz/seq must be numbers'), 400
        if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)):
            return jsonify(ok=False, error='vx/vy/vz must be finite'), 400
        if node.stop_requested():
            return jsonify(ok=False, error='soft-stopped; resume first'), 409
        if not node.cartesian_active():
            return jsonify(ok=False, error='not in joystick mode'), 409
        if not node_healthy():
            # The streamer/watchdog thread is dead: nothing safe can happen.
            node.soft_stop()
            return jsonify(
                ok=False,
                error='jog node unhealthy - soft-stopped; use the hardware E-stop '
                      'if the arm is still moving, then restart jog_ui',
            ), 503
        with lease_lock:
            if data.get('token') != lease['token'] or lease['token'] is None:
                return jsonify(ok=False, error='controlled by another client'), 409
            if seq <= lease['seq']:
                # Stale/reordered packet (e.g. delivered after the release-zero):
                # dropping it is what keeps 'release to stop' truthful.
                return jsonify(ok=True, stale=True)
            lease['seq'] = seq
            applied = node.set_twist(vx, vy, vz)  # clamped inside
        return jsonify(ok=True, applied_mps=applied)

    @app.post('/api/jog')
    def jog():
        if not origin_ok():
            return jsonify(ok=False, error='cross-origin request rejected'), 403
        # No force=True: requiring application/json makes browsers preflight
        # cross-origin requests, which fail here since no CORS headers are served.
        data = request.get_json(silent=True)
        if data is None:
            return jsonify(ok=False, error='expected an application/json body'), 400
        joint = data.get('joint')
        if isinstance(joint, bool) or not isinstance(joint, int):
            return jsonify(ok=False, error='joint must be an integer'), 400
        try:
            delta_deg = float(data.get('delta_deg'))
        except (TypeError, ValueError):
            return jsonify(ok=False, error='delta_deg must be a number'), 400
        # Reject rather than clamp: malformed input must never produce motion.
        if not math.isfinite(delta_deg) or delta_deg == 0.0 \
                or abs(delta_deg) > max_step_deg + 1e-9:
            return jsonify(
                ok=False, error='delta_deg must be finite, nonzero, |x| <= %g' % max_step_deg
            ), 400
        if node.stop_requested():
            return jsonify(ok=False, error='soft-stopped; resume first'), 409
        if node.cartesian_active():
            return jsonify(
                ok=False,
                error="joystick (cartesian) mode is active; POST /api/mode "
                      "{'mode': 'joint'} before per-joint jogs",
            ), 409
        if not busy.acquire(blocking=False):
            return jsonify(ok=False, error='a move is already in progress'), 409
        try:
            ok = node.nudge_joint(joint, delta_deg, step_time_s)
            return jsonify(ok=True) if ok else (jsonify(ok=False, error='move failed'), 500)
        finally:
            busy.release()

    @app.post('/api/home')
    def home():
        if not origin_ok():
            return jsonify(ok=False, error='cross-origin request rejected'), 403
        data = request.get_json(silent=True)
        if data is None or data.get('confirm') is not True:
            return jsonify(ok=False, error="expected application/json {'confirm': true}"), 400
        if node.stop_requested():
            return jsonify(ok=False, error='soft-stopped; resume first'), 409
        if node.cartesian_active():
            return jsonify(ok=False, error='disable the joystick first'), 409
        if not busy.acquire(blocking=False):
            return jsonify(ok=False, error='a move is already in progress'), 409
        try:
            ok = node.move_to_joint_positions(node.home_pose, node.home_time_s)
            return jsonify(ok=True) if ok else (
                jsonify(ok=False, error='home move failed'), 500
            )
        finally:
            busy.release()

    @app.post('/api/gripper')
    def gripper():
        if not origin_ok():
            return jsonify(ok=False, error='cross-origin request rejected'), 403
        data = request.get_json(silent=True)
        if data is None:
            return jsonify(ok=False, error='expected an application/json body'), 400
        action = data.get('action')
        if action not in ('open', 'close'):
            return jsonify(ok=False, error="action must be 'open' or 'close'"), 400
        if node.stop_requested():
            return jsonify(ok=False, error='soft-stopped; resume first'), 409
        if not busy.acquire(blocking=False):
            return jsonify(ok=False, error='a move is already in progress'), 409
        try:
            ok = node.open_gripper() if action == 'open' else node.close_gripper()
            return jsonify(ok=True) if ok else (
                jsonify(ok=False, error='gripper command failed'), 500
            )
        finally:
            busy.release()

    @app.post('/api/estop')
    def estop():
        # Deliberately the most permissive route: no busy gate, no origin check,
        # no body required — a stop must always go through, whoever asks.
        node.soft_stop()
        with lease_lock:
            lease['token'] = None
        return jsonify(ok=True)

    @app.post('/api/resume')
    def resume():
        if not origin_ok():
            return jsonify(ok=False, error='cross-origin request rejected'), 403
        data = request.get_json(silent=True)
        if data is None or data.get('confirm') is not True:
            return jsonify(ok=False, error="expected application/json {'confirm': true}"), 400
        if busy.locked():
            # Never un-stop while a motion is still in flight or being cancelled.
            return jsonify(ok=False, error='motion still settling; try again'), 409
        node.resume()
        return jsonify(ok=True)

    node.get_logger().info(
        'Jog UI on http://%s:%d (dry_run=%s)' % (ui_host, ui_port, node.dry_run)
    )
    skip_stop = False
    try:
        # Werkzeug 2.0's dev server swallows KeyboardInterrupt inside
        # serve_forever(), so app.run() returns NORMALLY on SIGINT. The stop
        # therefore lives in finally, which runs on EVERY exit path (including
        # exceptions Werkzeug does not swallow). (Werkzeug 2.0 serves HTTP/1.0
        # Connection: close; if ever upgraded to >=2.1, revisit browser
        # keep-alive POST auto-retry, which could double-fire a jog.)
        app.run(host=ui_host, port=ui_port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        # Bind failure (e.g. port already in use): do NOT soft-stop — another
        # jog_ui instance may be live, and deactivating the controller would
        # yank it out from under that instance mid-motion.
        skip_stop = True
        node.get_logger().fatal('UI server failed to start: %s' % exc)
    finally:
        if not skip_stop:
            # A second Ctrl-C must not abort the stop sequence.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            node.get_logger().warn('Server exiting; soft-stopping.')
            for _ in range(2):
                try:
                    node.soft_stop()
                    time.sleep(0.5)  # let the cancels / deactivation flush
                    break
                except BaseException:
                    continue
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
