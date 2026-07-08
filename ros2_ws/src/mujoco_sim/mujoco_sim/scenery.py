"""Scene dressing for build_scene: materials, lighting, styled furniture/props.

Everything here is VISUAL + local-physics polish; the planner's collision
world still comes from scene.yaml's primitive envelopes (planner_node builds
bounding boxes from the same YAML). Rule: styled geometry must stay INSIDE
its YAML envelope so the planner never under-covers a prop (one documented
exception: the mug handle, which points +x away from all approach corridors).

The world model: the arm is mounted on a TABLE standing in a room. Room floor
at FLOOR_Z, table legs down to it, arm on a pedestal at the origin. scene.yaml
z-coordinates are unchanged (base_link frame, tabletop at -0.07); previously
the floor plane sat at z=0 which buried the table and floated the props.

All mjSpec calls that vary across mujoco versions are guarded — a failure
prints a warning and falls back to plain geometry, never a broken scene.
"""

import math
import sys

FLOOR_Z = -0.75          # room floor in base_link coordinates (table height)
TABLE_TOP = -0.07        # matches scene.yaml's table box top face


def _e2q(rpy_deg):
    r, p, y = (math.radians(a) for a in rpy_deg)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    x = sr * cp * cy - cr * sp * sy
    yy = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return [w, x, yy, z]     # MuJoCo quat order


def _warn(msg):
    print('scenery WARNING: %s' % msg, file=sys.stderr)


# --------------------------------------------------------------------- assets
def add_assets(spec):
    """Skybox, floor checker, and named materials. Best-effort per item."""
    import mujoco
    try:
        spec.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.62, 0.70, 0.80], rgb2=[0.18, 0.22, 0.30],
                         width=512, height=3072)
    except Exception as exc:
        _warn('skybox texture failed (%s)' % exc)
    try:
        spec.add_texture(name='tex_floor', type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.31, 0.32, 0.34], rgb2=[0.38, 0.39, 0.42],
                         mark=mujoco.mjtMark.mjMARK_EDGE,
                         markrgb=[0.24, 0.25, 0.27], width=300, height=300)
    except Exception as exc:
        _warn('floor texture failed (%s)' % exc)

    def mat(name, rgba, specular=0.3, shininess=0.4, reflectance=0.0,
            texture=None, texrepeat=None):
        try:
            m = spec.add_material(name=name)
            m.rgba = rgba
            m.specular = specular
            m.shininess = shininess
            m.reflectance = reflectance
            if texrepeat is not None:
                m.texrepeat = texrepeat
            if texture is not None:
                try:
                    import mujoco as mj
                    m.textures[int(mj.mjtTextureRole.mjTEXROLE_RGB)] = texture
                except Exception:
                    # older mjSpec: single 'texture' string attribute
                    try:
                        m.texture = texture
                    except Exception as exc:
                        _warn('material %s texture binding failed (%s)' % (name, exc))
        except Exception as exc:
            _warn('material %s failed (%s)' % (name, exc))

    mat('mat_floor', [1, 1, 1, 1], specular=0.2, shininess=0.2,
        reflectance=0.12, texture='tex_floor', texrepeat=[10, 10])
    mat('mat_wood', [0.55, 0.39, 0.25, 1], specular=0.25, shininess=0.35)
    mat('mat_wood_dark', [0.33, 0.22, 0.14, 1], specular=0.2, shininess=0.3)
    mat('mat_wood_mid', [0.44, 0.30, 0.19, 1], specular=0.2, shininess=0.3)
    mat('mat_wall', [0.78, 0.77, 0.73, 1], specular=0.05, shininess=0.05)
    mat('mat_metal', [0.75, 0.77, 0.80, 1], specular=0.9, shininess=0.8,
        reflectance=0.25)
    mat('mat_pedestal', [0.22, 0.23, 0.25, 1], specular=0.6, shininess=0.6)
    mat('mat_counter', [0.35, 0.36, 0.39, 1], specular=0.55, shininess=0.6,
        reflectance=0.08)                      # stone countertop
    mat('mat_white', [0.91, 0.92, 0.94, 1], specular=0.5, shininess=0.5)
    mat('mat_dark', [0.17, 0.17, 0.19, 1], specular=0.4, shininess=0.5)


def set_visual(spec):
    """Renderer quality knobs (shadows, headlight); harmless if absent."""
    try:
        # 2048, not 4096: the Jetson's GPU also runs cuRobo — an expensive
        # shadowmap can slow the sim loop enough to make motion look ragged.
        spec.visual.quality.shadowsize = 2048
    except Exception as exc:
        _warn('shadowsize failed (%s)' % exc)
    try:
        g = getattr(spec.visual, 'global_', None) or spec.visual.global_
        g.offwidth = 1280
        g.offheight = 800
    except Exception as exc:
        _warn('offscreen framebuffer size failed (%s)' % exc)
    try:
        hl = spec.visual.headlight
        hl.ambient = [0.30, 0.30, 0.32]
        hl.diffuse = [0.35, 0.35, 0.35]
        hl.specular = [0.2, 0.2, 0.2]
    except Exception as exc:
        _warn('headlight failed (%s)' % exc)


def add_lights(world):
    """Warm key with shadows + cool fill + soft top."""
    # All lights live on the ROBOT side of the back wall (x < 0.78) — a light
    # behind the wall lights nothing but the wall's back.
    lights = [
        dict(name='key', pos=[-1.5, -1.3, 1.7], dir=[0.66, 0.44, -0.61],
             diffuse=[0.75, 0.72, 0.66], castshadow=True),
        dict(name='fill', pos=[-1.2, 1.5, 1.2], dir=[0.5, -0.6, -0.62],
             diffuse=[0.28, 0.30, 0.36], castshadow=False),
        dict(name='top', pos=[0.3, 0.0, 2.4], dir=[0, 0, -1],
             diffuse=[0.30, 0.30, 0.30], castshadow=False),
    ]
    for kw in lights:
        cast = kw.pop('castshadow')
        try:
            lt = world.add_light(**kw)
            try:
                lt.castshadow = cast
            except Exception:
                pass
        except Exception as exc:
            _warn('light %s failed (%s)' % (kw.get('name'), exc))
            break


# ------------------------------------------------------------------ furniture
def add_room(world, mujoco):
    """Floor at real room height + pedestal the arm stands on."""
    world.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[4.0, 4.0, 0.1], pos=[0, 0, FLOOR_Z],
                   material='mat_floor')
    # (The arm's pedestal is a scene.yaml obstacle now — rendered by
    # render_obstacle's 'pedestal' key, and modeled by the planner.)


def render_obstacle(world, o, mujoco):
    """Furniture, styled by name. The YAML box is ALWAYS rendered as-is (it is
    the exact planner collision volume); styling only ADDS support geometry
    below/around it (legs, plinths, wall extension) outside the arm's reach."""
    name = o['name']
    dims = [float(v) for v in o['dims']]
    pos = [float(v) for v in o['position']]
    half = [d / 2.0 for d in dims]
    top = pos[2] + half[2]
    bottom = pos[2] - half[2]

    def box(suffix, size, at, material):
        world.add_geom(name='obs_%s_%s' % (name, suffix),
                       type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=size, pos=at, material=material)

    if name == 'pedestal':
        # Full-height mounting column (YAML box stops at -0.02 so the
        # planner's copy stays clear of the base_link collision sphere).
        world.add_geom(name='obs_pedestal', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[min(half[0], half[1]), (0.0 - TABLE_TOP) / 2.0, 0],
                       pos=[pos[0], pos[1], TABLE_TOP / 2.0],
                       material='mat_pedestal')
    elif name == 'table':
        # Kitchen ISLAND: stone top on a solid wood base down to a kickboard.
        box('top', half, pos, 'mat_counter')
        base_h = (bottom - FLOOR_Z - 0.06) / 2.0
        box('base', [half[0] - 0.05, half[1] - 0.05, base_h],
            [pos[0], pos[1], bottom - base_h], 'mat_wood_mid')
        box('kick', [half[0] - 0.10, half[1] - 0.10, 0.03],
            [pos[0], pos[1], FLOOR_Z + 0.03], 'mat_dark')
    elif name == 'wall_counter':
        box('top', half, pos, 'mat_counter')
        base_h = (bottom - FLOOR_Z - 0.06) / 2.0
        box('base', [half[0] - 0.02, half[1] - 0.02, base_h],
            [pos[0] + 0.02, pos[1], bottom - base_h], 'mat_wood_mid')
        box('kick', [half[0] - 0.05, half[1] - 0.06, 0.03],
            [pos[0] + 0.05, pos[1], FLOOR_Z + 0.03], 'mat_dark')
    elif name == 'upper_cabinet':
        box('main', half, pos, 'mat_wood_mid')
        for tag, sy in (('l', -1), ('r', 1)):
            box('door_%s' % tag, [0.006, half[1] * 0.46, half[2] * 0.90],
                [pos[0] - half[0] - 0.006, pos[1] + sy * half[1] * 0.5, pos[2]],
                'mat_wood_dark')
            box('knob_%s' % tag, [0.008, 0.008, 0.008],
                [pos[0] - half[0] - 0.02, pos[1] + sy * 0.05, pos[2] - half[2] * 0.7],
                'mat_metal')
    elif name == 'microwave':
        box('body', half, pos, 'mat_dark')
        box('window', [0.004, half[1] * 0.55, half[2] * 0.62],
            [pos[0] - half[0] - 0.004, pos[1] + half[1] * 0.15, pos[2]],
            'mat_pedestal')
        box('handle', [0.008, 0.008, half[2] * 0.7],
            [pos[0] - half[0] - 0.012, pos[1] - half[1] + 0.03, pos[2]],
            'mat_metal')
    elif name == 'fridge':
        box('body', half, pos, 'mat_white')
        box('door_split', [half[0] + 0.002, half[1], 0.004],
            [pos[0], pos[1], pos[2] + half[2] * 0.25], 'mat_pedestal')
        for tag, dz in (('t', half[2] * 0.32), ('b', 0.10)):
            box('handle_%s' % tag, [0.012, 0.012, 0.10],
                [pos[0] - half[0] - 0.015, pos[1] - half[1] + 0.07, pos[2] + dz],
                'mat_metal')
    elif name == 'stool':
        box('seat', [half[0], half[1], 0.02], [pos[0], pos[1], top - 0.02],
            'mat_wood')
        leg_h = (top - 0.04 - FLOOR_Z) / 2.0
        for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            box('leg_%d' % i, [0.018, 0.018, leg_h],
                [pos[0] + sx * (half[0] - 0.03), pos[1] + sy * (half[1] - 0.03),
                 FLOOR_Z + leg_h], 'mat_wood_dark')
    elif name == 'trash_bin':
        r = min(half[0], half[1])
        world.add_geom(name='obs_trash_body', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[r, half[2], 0], pos=pos, material='mat_dark')
        world.add_geom(name='obs_trash_lip', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[r + 0.008, 0.012, 0], pos=[pos[0], pos[1], top - 0.012],
                       material='mat_pedestal')
    elif name == 'back_wall':
        box('main', half, pos, 'mat_wall')
        # Continue the wall down to the room floor.
        ext_h = (bottom - FLOOR_Z) / 2.0
        if ext_h > 0:
            box('lower', [half[0], half[1], ext_h],
                [pos[0], pos[1], FLOOR_Z + ext_h], 'mat_wall')
        # Baseboard.
        box('baseboard', [0.012, half[1], 0.045],
            [pos[0] - half[0] - 0.012, pos[1], FLOOR_Z + 0.045], 'mat_wood_dark')
    elif name == 'cabinet':
        box('main', half, pos, 'mat_wood_dark')
        # Lip stays flush with the collision envelope (no overhang: the
        # planner would not know about it).
        box('top_lip', [half[0], half[1], 0.006],
            [pos[0], pos[1], top - 0.006], 'mat_wood_mid')
        plinth_h = (bottom - TABLE_TOP) / 2.0
        if plinth_h > 0:
            box('plinth', [half[0] - 0.02, half[1] - 0.02, plinth_h],
                [pos[0], pos[1], TABLE_TOP + plinth_h], 'mat_wood_dark')
    elif name == 'shelf':
        # Just the slab — a front edge trim would poke outside the collision
        # envelope, exactly where the shelf_edge target parks the fingertips.
        box('slab', half, pos, 'mat_wood')
    else:
        color = [float(v) for v in o.get('color', [0.5, 0.5, 0.5, 1.0])]
        world.add_geom(name='obs_' + name, type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=half, pos=pos, rgba=color[:3] + [1.0])


# ---------------------------------------------------------------------- props
def _add_geom(parent, mujoco, name, gtype, size, pos, rgba=None, material=None,
              quat=None, density=None, visual_only=False):
    kw = dict(name=name, type=gtype, size=size, pos=pos)
    if rgba is not None:
        kw['rgba'] = rgba
    if material is not None:
        kw['material'] = material
    if quat is not None:
        kw['quat'] = quat
    if density is not None:
        kw['density'] = density
    if visual_only:
        kw['contype'] = 0
        kw['conaffinity'] = 0
    return parent.add_geom(**kw)


def render_object(world, o, mujoco):
    """Props, styled by name; free ones become free bodies with real physics.

    Positions inside a free body are LOCAL (body frame at the YAML pose);
    static props place geoms in world coordinates directly.
    """
    name = o['name']
    pos = [float(v) for v in o['position']]
    quat = _e2q([float(v) for v in o.get('rpy_deg', [0, 0, 0])])
    rgba = [float(v) for v in o.get('color', [0.8, 0.8, 0.8, 1.0])]
    density = float(o.get('density', 400.0))
    free = bool(o.get('free', False))

    if free:
        parent = world.add_body(name='obj_' + name, pos=pos, quat=quat)
        try:
            parent.add_freejoint()
        except AttributeError:
            parent.add_joint(name='obj_%s_free' % name,
                             type=mujoco.mjtJoint.mjJNT_FREE)
        origin = [0.0, 0.0, 0.0]
        gq = None
    else:
        parent = world
        origin = pos
        gq = quat

    def at(dx=0.0, dy=0.0, dz=0.0):
        return [origin[0] + dx, origin[1] + dy, origin[2] + dz]

    CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
    BOX = mujoco.mjtGeom.mjGEOM_BOX
    CAP = mujoco.mjtGeom.mjGEOM_CAPSULE
    ELL = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    g = lambda *a, **kw: _add_geom(parent, mujoco, *a, **kw)

    if name == 'bottle':
        r = float(o.get('radius', 0.033))
        h = float(o.get('height', 0.22))
        body_rgba = rgba[:3] + [0.72]                     # translucent plastic
        # body 62% / shoulder / neck / cap — total exactly h, all within r.
        g('obj_bottle_body', CYL, [r, 0.31 * h, 0], at(dz=-0.19 * h),
          rgba=body_rgba, density=density)
        g('obj_bottle_shoulder', ELL, [0.95 * r, 0.95 * r, 0.07 * h],
          at(dz=0.12 * h), rgba=body_rgba, density=density)
        g('obj_bottle_neck', CYL, [0.45 * r, 0.09 * h, 0], at(dz=0.27 * h),
          rgba=body_rgba, density=density)
        g('obj_bottle_cap', CYL, [0.55 * r, 0.07 * h, 0], at(dz=0.43 * h),
          rgba=[0.92, 0.94, 0.96, 1.0], density=density)
    elif name == 'mug':
        r = float(o.get('radius', 0.032))
        h = float(o.get('height', 0.09))
        g('obj_mug_body', CYL, [r, h / 2.0, 0], at(), rgba=rgba, density=density)
        g('obj_mug_inner', CYL, [0.8 * r, 0.002, 0], at(dz=h / 2.0 - 0.003),
          rgba=[0.15, 0.10, 0.08, 1.0], visual_only=True, density=100.0)
        # Handle points +x (toward the wall, AWAY from every approach corridor
        # — the planner's bounding cylinder does not cover it).
        hq = _e2q([0, 90, 0])
        g('obj_mug_handle_v', CAP, [0.0050, 0.020, 0], at(dx=r + 0.023),
          rgba=rgba, visual_only=True, density=100.0)
        g('obj_mug_handle_t', CAP, [0.0045, 0.011, 0], at(dx=r + 0.011, dz=0.021),
          rgba=rgba, quat=hq, visual_only=True, density=100.0)
        g('obj_mug_handle_b', CAP, [0.0045, 0.011, 0], at(dx=r + 0.011, dz=-0.021),
          rgba=rgba, quat=hq, visual_only=True, density=100.0)
    elif name == 'bowl':
        r = float(o.get('radius', 0.075))
        h = float(o.get('height', 0.05))
        g('obj_bowl_base', CYL, [0.55 * r, h / 4.0, 0], at(dz=-h / 4.0),
          rgba=rgba, density=density)
        g('obj_bowl_body', CYL, [r, h / 4.0, 0], at(dz=h / 4.0),
          rgba=rgba, density=density)
        g('obj_bowl_inner', CYL, [0.8 * r, 0.003, 0], at(dz=h / 2.0 - 0.004),
          rgba=[0.55, 0.42, 0.30, 1.0], visual_only=True, density=50.0)
    elif name == 'apple':
        r = float(o.get('radius', 0.035))
        g('obj_apple_body', mujoco.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], at(),
          rgba=rgba, density=density)
        g('obj_apple_stem', CAP, [0.003, 0.010, 0], at(dz=r + 0.008),
          rgba=[0.35, 0.25, 0.12, 1.0], visual_only=True, density=50.0)
    elif name == 'plate':
        r = float(o.get('radius', 0.09))
        h = float(o.get('height', 0.02))
        g('obj_plate_base', CYL, [0.55 * r, h / 4.0, 0], at(dz=-h / 4.0),
          rgba=rgba, density=density)
        g('obj_plate_top', CYL, [r, h / 4.0, 0], at(dz=h / 4.0),
          rgba=rgba, density=density)
    elif name == 'snack_box':
        dims = [float(v) for v in o.get('dims', [0.055, 0.14, 0.19])]
        half = [d / 2.0 for d in dims]
        g('obj_snack_box', BOX, half, at(), rgba=rgba, density=density)
        g('obj_snack_label', BOX, [0.0015, half[1] * 0.8, half[2] * 0.66],
          at(dx=half[0] + 0.0015), rgba=[0.95, 0.91, 0.80, 1.0],
          visual_only=True, density=50.0)
    elif name == 'cabinet_door':
        dims = [float(v) for v in o.get('dims', [0.015, 0.24, 0.44])]
        half = [d / 2.0 for d in dims]
        g('obj_door', BOX, half, at(), quat=gq, material='mat_wood_mid')
        g('obj_door_panel', BOX, [0.004, half[1] * 0.72, half[2] * 0.78],
          at(dx=-half[0] - 0.004), quat=gq, material='mat_wood_dark',
          visual_only=True)
    elif name == 'cabinet_handle_bar':
        dims = [float(v) for v in o.get('dims', [0.02, 0.025, 0.12])]
        half = [d / 2.0 for d in dims]
        g('obj_handle_bar', BOX, [0.008, 0.008, half[2]], at(), quat=gq,
          material='mat_metal')
        for tag, dz in (('t', half[2] - 0.015), ('b', -half[2] + 0.015)):
            g('obj_handle_nub_%s' % tag, BOX, [half[0], 0.006, 0.006],
              at(dx=half[0] - 0.004, dz=dz), quat=gq, material='mat_metal',
              visual_only=True)
    elif name.startswith('shelf_post'):
        dims = [float(v) for v in o.get('dims', [0.03, 0.03, 0.5])]
        g('obj_' + name, BOX, [d / 2.0 for d in dims], at(), quat=gq,
          material='mat_wood_mid')
    else:
        # Generic fallback: the plain primitive, exactly as before.
        otype = o.get('type', 'box')
        if otype == 'cylinder':
            gtype, size = CYL, [float(o.get('radius', 0.03)),
                                float(o.get('height', 0.1)) / 2.0, 0]
        elif otype == 'sphere':
            gtype, size = mujoco.mjtGeom.mjGEOM_SPHERE, \
                [float(o.get('radius', 0.03)), 0, 0]
        else:
            gtype, size = BOX, [float(v) / 2.0
                                for v in o.get('dims', [0.05, 0.05, 0.05])]
        g('obj_' + name + '_geom', gtype, size, at(), rgba=rgba, quat=gq,
          density=density)


# ----------------------------------------------------------------- entrypoint
def dress_world(spec, scene):
    """Everything above, in order. `scene` is the parsed scene.yaml dict."""
    import mujoco
    world = spec.worldbody
    add_assets(spec)
    set_visual(spec)
    add_lights(world)
    add_room(world, mujoco)
    # In FRONT of the wall (x < 0.78), on the robot's side, looking at the
    # tabletop scene. xyaxes = camera right + up for a look-at of
    # (0.45, 0, 0.15) from (-1.25, -0.95, 0.65).
    # Look-at of (0.30, 0.05, 0.15) from (-1.30, -1.00, 0.90) — the whole
    # island with the kitchen wall behind it.
    world.add_camera(name='viz_cam', pos=[-1.30, -1.00, 0.90],
                     xyaxes=[0.549, -0.836, 0.0, 0.305, 0.200, 0.931])
    for o in scene.get('obstacles', []):
        render_obstacle(world, o, mujoco)
    for o in scene.get('objects', []):
        render_object(world, o, mujoco)
