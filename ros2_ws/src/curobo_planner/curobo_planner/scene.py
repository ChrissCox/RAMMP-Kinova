"""Scene = single source of truth for obstacles + named targets.

Deliberately dependency-light (stdlib + PyYAML + ROS msgs only, NO cuRobo, NO
numpy) so the natural-language CLI can import it without the GPU stack. The
planner node consumes the same scene to build cuRobo's WorldConfig, guaranteeing
the obstacles the planner avoids are exactly the ones drawn in Foxglove.

Poses are authored human-friendly as position [x,y,z] (metres, base frame) plus
orientation as roll/pitch/yaw in DEGREES, converted to quaternions here.
"""

import math

import yaml

from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def euler_deg_to_quat(rpy_deg):
    """roll/pitch/yaw (degrees) -> (x, y, z, w) quaternion, ROS/xyzw order."""
    r, p, y = (math.radians(a) for a in rpy_deg)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,  # w
    )


class Obstacle:
    __slots__ = ('name', 'position', 'rpy_deg', 'dims', 'color')

    def __init__(self, d):
        self.name = d['name']
        self.position = [float(v) for v in d['position']]
        self.rpy_deg = [float(v) for v in d.get('rpy_deg', [0, 0, 0])]
        self.dims = [float(v) for v in d['dims']]
        self.color = [float(v) for v in d.get('color', [0.55, 0.4, 0.3, 0.85])]


class SceneObject:
    """A prop (bottle, mug, door...). Rendered in MuJoCo and Foxglove AND
    collision-avoided by the planner (as its bounding box) — except for the
    props a target names in its `ignore_objects` (you reach FOR the bottle,
    you can't also dodge it). Types: box (dims=full extents), cylinder
    (radius+height), sphere (radius). free=true gives it a free joint in
    MuJoCo (real physics: it can rest on, and be knocked off, the furniture)."""
    __slots__ = ('name', 'type', 'position', 'rpy_deg', 'dims', 'radius',
                 'height', 'color', 'free', 'density')

    def __init__(self, d):
        self.name = d['name']
        self.type = d.get('type', 'box')
        self.position = [float(v) for v in d['position']]
        self.rpy_deg = [float(v) for v in d.get('rpy_deg', [0, 0, 0])]
        self.dims = [float(v) for v in d.get('dims', [0.05, 0.05, 0.05])]
        self.radius = float(d.get('radius', 0.03))
        self.height = float(d.get('height', 0.1))
        self.color = [float(v) for v in d.get('color', [0.8, 0.8, 0.8, 1.0])]
        self.free = bool(d.get('free', False))
        self.density = float(d.get('density', 400.0))

    def bounding_dims(self):
        """Axis-aligned bounding box (full extents) — the collision proxy."""
        if self.type == 'cylinder':
            return [2 * self.radius, 2 * self.radius, self.height]
        if self.type == 'sphere':
            return [2 * self.radius] * 3
        return list(self.dims)


class Target:
    __slots__ = ('name', 'position', 'rpy_deg', 'keywords', 'description',
                 'ignore_objects', 'standoff', 'standoff_position',
                 'standoff_rpy_deg')

    def __init__(self, d):
        self.name = d['name']
        self.position = [float(v) for v in d['position']]
        self.rpy_deg = [float(v) for v in d.get('rpy_deg', [180, 0, 0])]
        self.keywords = [str(k).lower() for k in d.get('keywords', [])]
        self.description = str(d.get('description', ''))
        # Props to EXCLUDE from the collision world for the FINAL approach
        # segment only: the object being reached for must not be an obstacle
        # in the last few cm. The planner reaches a STANDOFF pose with the
        # FULL world first, and retreats through it on departure — the prop
        # is never exempt in transit. The standoff defaults to `standoff`
        # metres back along the tool axis; targets whose approach ray exits
        # the reachable workspace override it with an explicit
        # standoff_position (+ optional standoff_rpy_deg, e.g. a steeper
        # pre-grasp pitch above the object).
        self.ignore_objects = [str(n) for n in d.get('ignore_objects', [])]
        self.standoff = float(d.get('standoff', 0.10))
        sp = d.get('standoff_position')
        self.standoff_position = [float(v) for v in sp] if sp else None
        sr = d.get('standoff_rpy_deg')
        self.standoff_rpy_deg = [float(v) for v in sr] if sr else None

    def quat_xyzw(self):
        return euler_deg_to_quat(self.rpy_deg)


class Scene:
    def __init__(self, base_frame, obstacles, targets, objects=()):
        self.base_frame = base_frame
        self.obstacles = obstacles
        self.targets = targets
        self.objects = list(objects)

    @property
    def target_names(self):
        return [t.name for t in self.targets]

    def target(self, name):
        for t in self.targets:
            if t.name == name:
                return t
        return None


def load_scene(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return Scene(
        base_frame=data.get('base_frame', 'base_link'),
        obstacles=[Obstacle(o) for o in data.get('obstacles', [])],
        targets=[Target(t) for t in data.get('targets', [])],
        objects=[SceneObject(o) for o in data.get('objects', [])],
    )


def _quat_msg(xyzw):
    q = Quaternion()
    q.x, q.y, q.z, q.w = xyzw
    return q


def scene_markers(scene, goal_target_name=None, stamp=None):
    """MarkerArray: obstacles (cubes), targets (spheres + labels), optional goal.

    Starts with a DELETEALL so live-removing an entry from scene.yaml also
    removes its marker — the display must never show an obstacle the planner
    no longer avoids.
    """
    arr = MarkerArray()
    wipe = Marker()
    wipe.header.frame_id = scene.base_frame
    if stamp is not None:
        wipe.header.stamp = stamp
    wipe.action = Marker.DELETEALL
    arr.markers.append(wipe)
    mid = 0

    def _base(ns, mtype):
        nonlocal mid
        m = Marker()
        m.header.frame_id = scene.base_frame
        if stamp is not None:
            m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        mid += 1
        m.type = mtype
        m.action = Marker.ADD
        return m

    for o in scene.obstacles:
        m = _base('obstacles', Marker.CUBE)
        m.pose = Pose(position=Point(x=o.position[0], y=o.position[1], z=o.position[2]),
                      orientation=_quat_msg(euler_deg_to_quat(o.rpy_deg)))
        m.scale = Vector3(x=o.dims[0], y=o.dims[1], z=o.dims[2])
        m.color = ColorRGBA(r=o.color[0], g=o.color[1], b=o.color[2], a=o.color[3])
        arr.markers.append(m)

    for o in scene.objects:
        if o.type == 'cylinder':
            m = _base('objects', Marker.CYLINDER)
            m.scale = Vector3(x=2 * o.radius, y=2 * o.radius, z=o.height)
        elif o.type == 'sphere':
            m = _base('objects', Marker.SPHERE)
            m.scale = Vector3(x=2 * o.radius, y=2 * o.radius, z=2 * o.radius)
        else:
            m = _base('objects', Marker.CUBE)
            m.scale = Vector3(x=o.dims[0], y=o.dims[1], z=o.dims[2])
        m.pose = Pose(position=Point(x=o.position[0], y=o.position[1], z=o.position[2]),
                      orientation=_quat_msg(euler_deg_to_quat(o.rpy_deg)))
        m.color = ColorRGBA(r=o.color[0], g=o.color[1], b=o.color[2], a=o.color[3])
        arr.markers.append(m)

    for t in scene.targets:
        is_goal = (t.name == goal_target_name)
        s = _base('targets', Marker.SPHERE)
        s.pose = Pose(position=Point(x=t.position[0], y=t.position[1], z=t.position[2]),
                      orientation=_quat_msg((0.0, 0.0, 0.0, 1.0)))
        r = 0.05 if is_goal else 0.03
        s.scale = Vector3(x=r, y=r, z=r)
        if is_goal:
            s.color = ColorRGBA(r=0.2, g=1.0, b=0.3, a=1.0)
        else:
            s.color = ColorRGBA(r=0.4, g=0.7, b=1.0, a=0.9)
        arr.markers.append(s)

        label = _base('target_labels', Marker.TEXT_VIEW_FACING)
        label.pose = Pose(
            position=Point(x=t.position[0], y=t.position[1], z=t.position[2] + 0.06),
            orientation=_quat_msg((0.0, 0.0, 0.0, 1.0)))
        label.scale = Vector3(x=0.0, y=0.0, z=0.04)
        label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
        label.text = t.name
        arr.markers.append(label)

    return arr
