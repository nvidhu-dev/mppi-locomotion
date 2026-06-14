"""Unitree Go2 locomotion environment for MPPI.

Unlike the DM-Control ``suite`` walker, the Go2 ships only as
a raw MuJoCo model in `MuJoCo Menagerie
<https://github.com/google-deepmind/mujoco_menagerie>`_ (fetched here via the
``robot_descriptions`` package) — there is **no task and no reward**. This module
supplies both:

* :func:`make_env` builds a ``dm_control`` :class:`~dm_control.rl.control.Environment`
  from the Menagerie ``scene.xml`` (robot + ground plane + lights) with a chase
  camera added so the robot stays in frame for long recordings.
* :class:`Go2Locomotion` is a forward-walking task: a reward that wants the base
  upright, at standing height, and moving forward — exactly the kind of objective
  the MPPI planner optimizes over the true simulator.

The raw Menagerie model ships with 12 **torque** motors, but a real Unitree Go2 is
commanded with joint *position* targets tracked by an onboard PD loop — and that
is far more stable for sampling MPC. So :func:`_make_physics` converts the motors
to position (PD) actuators, and the planner samples small target offsets around
the standing pose (exposed as the task's ``nominal_action``) instead of around an
all-zero, collapsed configuration.
"""

import collections
import os

import numpy as np
from dm_control import mjcf
from dm_control.rl import control
from dm_control.suite import base
from dm_control.utils import rewards
from robot_descriptions import go2_mj_description

# Physics integrates at 2 ms; the planner/controller acts every 20 ms (10 substeps).
_CONTROL_TIMESTEP = 0.02
_DEFAULT_TIME_LIMIT = 25.0  # seconds

# Target forward speed (m/s) per task. 0.0 means "just stand".
_SPEEDS = {'stand': 0.0, 'walk': 1.0, 'run': 2.0}

_STAND_HEIGHT = 0.22  # base height (m) above which the robot counts as standing

# PD gains for the position actuators (Nm/rad and Nm*s/rad). Onboard-controller
# scale for the Go2; firm enough to hold the stance, soft enough to step.
_KP, _KV = 30.0, 1.0


def _make_physics():
    """Build Physics from Menagerie's go2 scene, with PD position actuators.

    Returns ``(physics, home_targets)`` where ``home_targets`` are the 12 standing
    joint angles used as the planner's nominal/rest action.
    """
    scene = os.path.join(os.path.dirname(go2_mj_description.MJCF_PATH), 'scene.xml')
    model = mjcf.from_path(scene)

    # A COM-tracking camera so the robot stays centered as it walks off.
    model.worldbody.add('camera', name='track', mode='trackcom',
                        pos=[0, -2.0, 0.7], xyaxes=[1, 0, 0, 0, 0.4, 1])

    # Replace the torque motors with position (PD) actuators, target = joint angle.
    # The joint ranges and motor torque limits live in default classes, so read
    # them off a compiled snapshot, then clamp each target to the joint range and
    # cap PD force at the original motor limit (respect the real torque budget).
    motors = model.find_all('actuator')
    joints = [a.joint for a in motors]
    snap = mjcf.Physics.from_mjcf_model(model)
    joint_ranges = [snap.named.model.jnt_range[j.name].copy() for j in joints]
    torque_limits = [snap.named.model.actuator_ctrlrange[a.name].copy() for a in motors]

    for a in motors:
        a.remove()
    for j, jrange, tlim in zip(joints, joint_ranges, torque_limits):
        model.actuator.add('position', name=j.name, joint=j, kp=_KP, kv=_KV,
                           ctrlrange=jrange, forcerange=tlim)

    physics = mjcf.Physics.from_mjcf_model(model)
    home_targets = physics.model.key_qpos[0][7:].copy()  # standing joint angles
    return physics, home_targets


class Go2Locomotion(base.Task):
    """Keep the Go2 upright and standing tall; optionally move it forward."""

    def __init__(self, move_speed, random=None):
        self._move_speed = move_speed
        super().__init__(random=random)

    def initialize_episode(self, physics):
        # Start from the standing 'home' keyframe with a little joint jitter.
        with physics.reset_context():
            physics.data.qpos[:] = physics.model.key_qpos[0]
            physics.data.qvel[:] = 0.0
            n_joints = physics.model.nq - 7  # minus the free-floating base pose
            physics.data.qpos[7:] += 0.03 * self.random.randn(n_joints)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['joints'] = physics.data.qpos[7:].copy()       # 12 leg joint angles
        obs['velocity'] = physics.data.qvel.copy()         # base + joint velocities
        obs['height'] = np.array([physics.named.data.xpos['base'][2]])
        return obs

    def get_reward(self, physics):
        nd = physics.named.data
        height = nd.xpos['base'][2]
        torso_up = nd.xmat['base'][8]      # base z-axis . world up, in [-1, 1]
        forward_vel = physics.data.qvel[0]  # world-frame x velocity of the base
        lateral_vel = physics.data.qvel[1]  # world-frame y velocity (sideways drift)
        heading_y = nd.xmat['base'][3]      # base x-axis, world-y component (yaw off +x)

        # Upright and at standing height -> the "alive" reward, in [0, 1].
        upright = rewards.tolerance(torso_up, bounds=(0.9, float('inf')),
                                    sigmoid='linear', margin=1.9, value_at_margin=0.0)
        standing = rewards.tolerance(height, bounds=(_STAND_HEIGHT, float('inf')),
                                     sigmoid='linear', margin=_STAND_HEIGHT,
                                     value_at_margin=0.0)
        stand_reward = upright * standing

        if self._move_speed == 0.0:
            return stand_reward

        # Forward-speed bonus, gated by being upright so it can't be earned by diving.
        move = rewards.tolerance(forward_vel,
                                 bounds=(self._move_speed, float('inf')),
                                 margin=self._move_speed, value_at_margin=0.0,
                                 sigmoid='linear')
        # Go *straight*: discourage sideways drift and yawing off the +x heading,
        # otherwise the planner can crab sideways and still score forward velocity.
        no_drift = rewards.tolerance(lateral_vel, bounds=(-0.2, 0.2),
                                     sigmoid='linear', margin=0.6, value_at_margin=0.1)
        facing = rewards.tolerance(heading_y, bounds=(-0.15, 0.15),
                                   sigmoid='linear', margin=0.6, value_at_margin=0.1)
        straight = no_drift * facing
        return stand_reward * straight * (5.0 * move + 1.0) / 6.0


def make_env(task='walk', seed=0, time_limit=_DEFAULT_TIME_LIMIT):
    """Return a Go2 ``control.Environment`` for the given task ('stand'/'walk'/'run')."""
    if task not in _SPEEDS:
        valid = ', '.join(_SPEEDS)
        raise ValueError(f"unknown go2 task {task!r}; choose one of: {valid}")
    physics, home_targets = _make_physics()
    task_obj = Go2Locomotion(move_speed=_SPEEDS[task],
                             random=np.random.RandomState(seed))
    # The planner reads this to warm-start / refill its nominal around the stance.
    task_obj.nominal_action = home_targets
    return control.Environment(physics, task_obj, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP)
