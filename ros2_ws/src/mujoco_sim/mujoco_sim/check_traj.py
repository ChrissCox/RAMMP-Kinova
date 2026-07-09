"""Replay published trajectories against the MuJoCo model and name every hit.

Subscribes to the joint_trajectory_controller topic, steps each trajectory
KINEMATICALLY (set qpos, run collision detection — no dynamics), and prints
which robot geometry touches which scene geometry at which waypoint. Turns
"it's running into stuff" into "left_outer_finger vs obs_shelf_slab at
waypoints 41-47, max depth 8 mm".

Run alongside the bringup + planner on the Jetson:

    ros2 run mujoco_sim check_traj
    # or with a non-default model path:
    ros2 run mujoco_sim check_traj --ros-args -p mujoco_model:=/path/scene.xml

Free props are held at their spawn poses (their planner poses); prop-vs-
furniture resting contacts are ignored — only robot-vs-anything is reported.
Penetrations shallower than `min_depth_mm` (default 1 mm) are noise from the
collision margin and are ignored.
"""

import glob
import json
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


class TrajChecker(Node):

    def __init__(self):
        super().__init__('check_traj')
        import mujoco
        self._mujoco = mujoco

        model_path = self.declare_parameter(
            'mujoco_model',
            os.path.expanduser('~/.ros/mujoco_sim/scene_gen3.xml')).value
        topic = self.declare_parameter(
            'trajectory_topic',
            '/joint_trajectory_controller/joint_trajectory').value
        self.min_depth_mm = float(
            self.declare_parameter('min_depth_mm', 1.0).value)

        self._model = mujoco.MjModel.from_xml_path(model_path)
        # Generate near-contacts within 3 cm so clean trajectories can report
        # their CLOSEST APPROACH — "skimming but legal" vs "actually touching"
        # becomes a number. Kinematic checking only; dynamics never run here.
        self._model.geom_margin[:] = 0.03
        self._data = mujoco.MjData(self._model)

        # Robot bodies = everything hanging off the arm's root subtree (the
        # body that owns joint_1), so gripper links count as robot too.
        j1 = self._model.joint('joint_1')
        root = self._model.jnt_bodyid[j1.id]
        while self._model.body_parentid[root] != 0:   # walk up to worldbody child
            root = self._model.body_parentid[root]
        self._robot_bodies = {
            b for b in range(self._model.nbody) if self._is_under(b, root)}

        # Record every trajectory for post-mortem physics replay: waypoint
        # checks can be clean while the EXECUTED physics deflects (contact
        # mid-motion) — the recording is the ground truth for that gap.
        self._log_dir = os.path.expanduser('~/.ros/mujoco_sim/traj_log')
        os.makedirs(self._log_dir, exist_ok=True)
        self._seq = 0

        self.create_subscription(JointTrajectory, topic, self._traj_cb, 10)

        # EXECUTION truth monitor: planned waypoints can be clean while the
        # controller deviates (or a display lies). Watch the ACTUAL streamed
        # joint states in a second MjData and report any real robot-scene
        # contact the moment it exists.
        self._exec_data = self._mujoco.MjData(self._model)
        self._exec_last_warn = {}
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)

        self.get_logger().info(
            'Checking trajectories from %s against %s (%d robot bodies); '
            'recording to %s; live execution monitor on /joint_states'
            % (topic, model_path, len(self._robot_bodies), self._log_dir))

    def _js_cb(self, msg):
        mujoco = self._mujoco
        model, data = self._model, self._exec_data
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            for n in ['joint_%d' % i for i in range(1, 8)]:
                data.qpos[model.jnt_qposadr[model.joint(n).id]] = \
                    msg.position[idx[n]]
        except (KeyError, IndexError):
            return
        mujoco.mj_forward(model, data)
        now = time.monotonic()
        for c in range(data.ncon):
            con = data.contact[c]
            depth = -float(con.dist)
            if depth < 0.002:
                continue
            b1 = model.geom_bodyid[con.geom1]
            b2 = model.geom_bodyid[con.geom2]
            r1, r2 = b1 in self._robot_bodies, b2 in self._robot_bodies
            if r1 == r2:
                continue
            key = (con.geom1, con.geom2)
            if now - self._exec_last_warn.get(key, 0.0) < 2.0:
                continue
            self._exec_last_warn[key] = now
            self.get_logger().error(
                'EXECUTION CONTACT (real, right now): %s vs %s, depth %.1f mm'
                % (self._geom_label(con.geom1), self._geom_label(con.geom2),
                   depth * 1000.0))

    def _record(self, msg):
        try:
            self._seq += 1
            out = {
                'stamp': self.get_clock().now().nanoseconds,
                'joint_names': list(msg.joint_names),
                'points': [{
                    'positions': list(p.positions),
                    'velocities': list(p.velocities),
                    't': p.time_from_start.sec + p.time_from_start.nanosec * 1e-9,
                } for p in msg.points],
            }
            path = os.path.join(self._log_dir, 'traj_%04d.json' % self._seq)
            with open(path, 'w') as f:
                json.dump(out, f)
            old = sorted(glob.glob(os.path.join(self._log_dir, 'traj_*.json')))
            for stale in old[:-50]:
                os.remove(stale)
        except Exception as exc:
            self.get_logger().warning('trajectory recording failed: %s' % exc)

    def _is_under(self, body, root):
        while body != 0:
            if body == root:
                return True
            body = self._model.body_parentid[body]
        return False

    def _geom_label(self, geom_id):
        name = self._mujoco.mj_id2name(
            self._model, self._mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name:
            return name
        body = self._model.body(self._model.geom_bodyid[geom_id]).name
        return '%s/geom%d' % (body, geom_id)

    def _traj_cb(self, msg):
        self._record(msg)
        mujoco = self._mujoco
        model, data = self._model, self._data
        try:
            qadr = [model.jnt_qposadr[model.joint(n).id] for n in msg.joint_names]
        except KeyError as exc:
            self.get_logger().error('Unknown joint in trajectory: %s' % exc)
            return

        # (pair label) -> [first_wp, last_wp, max_depth_m]
        hits = {}
        closest = [1e9, '', -1]   # min signed distance, pair, waypoint
        for k, pt in enumerate(msg.points):
            for adr, q in zip(qadr, pt.positions):
                data.qpos[adr] = q
            mujoco.mj_forward(model, data)   # runs collision detection
            for c in range(data.ncon):
                con = data.contact[c]
                b1 = model.geom_bodyid[con.geom1]
                b2 = model.geom_bodyid[con.geom2]
                r1 = b1 in self._robot_bodies
                r2 = b2 in self._robot_bodies
                if r1 == r2:
                    continue  # robot self-contact or scene-vs-scene resting
                robot_g = con.geom1 if r1 else con.geom2
                other_g = con.geom2 if r1 else con.geom1
                key = '%s  vs  %s' % (self._geom_label(robot_g),
                                      self._geom_label(other_g))
                dist = float(con.dist)       # <0 = penetration (margin: near-
                if dist < closest[0]:        # contacts appear with dist > 0)
                    closest[:] = [dist, key, k]
                depth = -dist
                if depth * 1000.0 < self.min_depth_mm:
                    continue
                if key not in hits:
                    hits[key] = [k, k, depth]
                else:
                    hits[key][1] = k
                    hits[key][2] = max(hits[key][2], depth)

        n = len(msg.points)
        if not hits:
            if closest[2] >= 0:
                self.get_logger().info(
                    'Trajectory (%d pts): clean; closest approach %.1f mm '
                    '(%s, wp %d).' % (n, closest[0] * 1000.0, closest[1],
                                      closest[2]))
            else:
                self.get_logger().info('Trajectory (%d pts): clean; nothing '
                                       'within 30 mm.' % n)
            return
        self.get_logger().warning('Trajectory (%d pts): %d contact pair(s):'
                                  % (n, len(hits)))
        for key, (k0, k1, depth) in sorted(hits.items(),
                                           key=lambda kv: -kv[1][2]):
            self.get_logger().warning(
                '  %s  waypoints %d-%d  max depth %.1f mm'
                % (key, k0, k1, depth * 1000.0))


def main(args=None):
    rclpy.init(args=args)
    node = TrajChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
