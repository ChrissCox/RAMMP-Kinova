"""Browser jog UI: move the arm with buttons instead of hard-coded motions.

Serves a single-page web UI (Flask, default port 8080) with per-joint nudge
buttons, gripper open/close, live joint angles, and a soft-stop button. Every
command goes through :class:`KinovaPrimitives` — same step clamp
(``max_nudge_deg``), same ``dry_run`` gate (defaults TRUE: nothing moves),
same soft-stop path as ``test_arm``.

SAFETY: the web soft-stop is a convenience, NOT a substitute for the hardware
E-stop. There is no authentication — anyone who can reach the port can move
the arm (motion routes do require an application/json body and a same-host
Origin, which blocks drive-by cross-site pages, but not a LAN attacker with
curl). Keep it on the robot LAN (or set ``ui_host`` to 127.0.0.1).
"""

import math
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
  body { font-family: system-ui, sans-serif; background: #14161a; color: #e8e8e8;
         max-width: 640px; margin: 0 auto; padding: 12px; }
  h1 { font-size: 1.2rem; margin: 8px 0; }
  #banner { padding: 8px 12px; border-radius: 6px; margin-bottom: 10px; display: none; }
  #banner.dry { display: block; background: #7a5b00; }
  #banner.stopped { display: block; background: #7a1f1f; }
  button { touch-action: manipulation; user-select: none; -webkit-user-select: none; }
  #stop { width: 100%; padding: 16px; font-size: 1.3rem; font-weight: bold;
          background: #c62828; color: white; border: none; border-radius: 8px;
          cursor: pointer; margin-bottom: 12px; }
  #resume { width: 100%; padding: 10px; font-size: 1rem; background: #2e7d32;
            color: white; border: none; border-radius: 8px; cursor: pointer;
            margin-bottom: 12px; }
  .row { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
  .row .name { flex: 1; }
  .row .angle { width: 72px; text-align: right; font-variant-numeric: tabular-nums;
                color: #9fd3ff; }
  button.jog, button.grip { padding: 10px 18px; font-size: 1.05rem; border: none;
          border-radius: 6px; background: #37474f; color: white; cursor: pointer; }
  button.jog:disabled, button.grip:disabled { opacity: 0.4; cursor: default; }
  select { padding: 6px; background: #22262c; color: #e8e8e8; border-radius: 6px; }
  #msg { min-height: 1.2em; color: #ffb74d; margin-top: 10px; font-weight: bold; }
  .grippers { display: flex; gap: 8px; margin-top: 12px; }
  .grippers button { flex: 1; }
</style>
</head>
<body>
<h1>RAMMP — Kinova jog</h1>
<div id="banner"></div>
<button id="stop">SOFT-STOP</button>
<button id="resume">Resume / reactivate controller</button>
<div class="row">
  <span class="name">Step size</span>
  <select id="step"></select>
</div>
<div id="joints"></div>
<div class="grippers">
  <button class="grip" id="gopen">Gripper open</button>
  <button class="grip" id="gclose">Gripper close</button>
</div>
<div id="msg"></div>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const esc = (t) => String(t).replace(/[&<>"']/g, (c) => "&#" + c.charCodeAt(0) + ";");
let built = false, inflight = 0, stopping = false, lostConn = false, lastStatus = null;

async function api(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
  return j;
}

function setMsg(text) { $("msg").textContent = text || ""; }

function setControlsDisabled(d) {
  document.querySelectorAll("button.jog, button.grip").forEach((b) => (b.disabled = d));
}

function build(s) {
  let steps = [0.5, 1, 2, 5, 10].filter((v) => v <= s.max_step_deg);
  if (!steps.length || steps[steps.length - 1] < s.max_step_deg) steps.push(s.max_step_deg);
  // Smallest step preselected: the conservative default for a jog panel.
  $("step").innerHTML = steps
    .map((v, i) => `<option value="${v}" ${i === 0 ? "selected" : ""}>${v}&deg;</option>`)
    .join("");
  $("joints").innerHTML = s.joint_names.map((n, i) => `
    <div class="row">
      <button class="jog" data-j="${i}" data-s="-1">&minus;</button>
      <span class="name">${esc(n)}</span>
      <span class="angle" id="a${i}">&mdash;</span>
      <button class="jog" data-j="${i}" data-s="1">+</button>
    </div>`).join("");
  document.querySelectorAll("button.jog").forEach((b) => {
    b.addEventListener("click", () => jog(+b.dataset.j, +b.dataset.s));
  });
  built = true;
}

function render(s) {
  lastStatus = s;
  if (!built) build(s);
  const banner = $("banner");
  if (s.stopped) {
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
  if (s.positions_deg) {
    s.positions_deg.forEach((p, i) => {
      const el = $("a" + i);
      if (el) el.textContent = p.toFixed(1) + "\\u00b0";
    });
  }
}

async function jog(joint, sign) {
  if (inflight) return;
  inflight++;
  setControlsDisabled(true);
  try {
    setMsg("moving " + $("step").value + "\\u00b0 ...");
    await api("/api/jog", {joint: joint, delta_deg: sign * parseFloat($("step").value)});
    setMsg("");
  } catch (e) { setMsg(e.message); }
  finally { inflight--; }
}

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

async function stopArm() {
  // Retry hard: a soft-stop lost to a WiFi hiccup must not fail silently.
  if (stopping) return;
  stopping = true;
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

$("stop").addEventListener("click", stopArm);
$("resume").addEventListener("click", async () => {
  try { await api("/api/resume", {confirm: true}); setMsg(""); }
  catch (e) { setMsg(e.message); }
});
$("gopen").addEventListener("click", () => grip("open"));
$("gclose").addEventListener("click", () => grip("close"));

async function poll() {
  try {
    const s = await (await fetch("/api/status")).json();
    if (lostConn) { lostConn = false; setMsg(""); }
    render(s);
  } catch (e) {
    lostConn = true;
    setMsg("connection lost");
  }
  setTimeout(poll, 500);
}
poll();
</script>
</body>
</html>
"""


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
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    busy = threading.Lock()  # one motion command at a time
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
            'max_step_deg': max_step_deg,
        })

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
        if not busy.acquire(blocking=False):
            return jsonify(ok=False, error='a move is already in progress'), 409
        try:
            ok = node.nudge_joint(joint, delta_deg, step_time_s)
            return jsonify(ok=True) if ok else (jsonify(ok=False, error='move failed'), 500)
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
    try:
        # Werkzeug 2.0's dev server swallows KeyboardInterrupt inside
        # serve_forever(), so app.run() returns NORMALLY on SIGINT — the
        # soft-stop must live on the normal-return path (else:), not only in
        # an exception handler. (Werkzeug 2.0 serves HTTP/1.0 Connection:
        # close; if ever upgraded to >=2.1, revisit browser keep-alive POST
        # auto-retry, which could double-fire a jog.)
        app.run(host=ui_host, port=ui_port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        node.get_logger().warn('Interrupted; soft-stopping.')
        node.soft_stop()
        time.sleep(0.5)  # let the cancels / deactivation request flush
    except OSError as exc:
        # Bind failure (e.g. port already in use): do NOT soft-stop — another
        # jog_ui instance may be live, and deactivating the controller would
        # yank it out from under that instance mid-motion.
        node.get_logger().fatal('UI server failed to start: %s' % exc)
    else:
        node.get_logger().warn('Server exited; soft-stopping.')
        node.soft_stop()
        time.sleep(0.5)
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
