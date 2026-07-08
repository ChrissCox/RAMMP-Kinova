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

import os

import rclpy
from rclpy.node import Node
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
        self._data = mujoco.MjData(self._model)

        # Robot bodies = everything hanging off the arm's root subtree (the
        # body that owns joint_1), so gripper links count as robot too.
        j1 = self._model.joint('joint_1')
        root = self._model.jnt_bodyid[j1.id]
        while self._model.body_parentid[root] != 0:   # walk up to worldbody child
            root = self._model.body_parentid[root]
        self._robot_bodies = {
            b for b in range(self._model.nbody) if self._is_under(b, root)}

        self.create_subscription(JointTrajectory, topic, self._traj_cb, 10)
        self.get_logger().info(
            'Checking trajectories from %s against %s (%d robot bodies)'
            % (topic, model_path, len(self._robot_bodies)))

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
        mujoco = self._mujoco
        model, data = self._model, self._data
        try:
            qadr = [model.jnt_qposadr[model.joint(n).id] for n in msg.joint_names]
        except KeyError as exc:
            self.get_logger().error('Unknown joint in trajectory: %s' % exc)
            return

        # (pair label) -> [first_wp, last_wp, max_depth_m]
        hits = {}
        for k, pt in enumerate(msg.points):
            for adr, q in zip(qadr, pt.positions):
                data.qpos[adr] = q
            mujoco.mj_forward(model, data)   # runs collision detection
            for c in range(data.ncon):
                con = data.contact[c]
                depth = -float(con.dist)     # positive = penetration
                if depth * 1000.0 < self.min_depth_mm:
                    continue
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
                if key not in hits:
                    hits[key] = [k, k, depth]
                else:
                    hits[key][1] = k
                    hits[key][2] = max(hits[key][2], depth)

        n = len(msg.points)
        if not hits:
            self.get_logger().info('Trajectory (%d pts): NO robot-scene '
                                   'contact. Clean.' % n)
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
