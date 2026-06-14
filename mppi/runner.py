"""Episode runner: load a DM-Control task, drive it with MPPI, optionally record.

This is the body-agnostic equivalent of the original ``run_mppi`` function — the
``domain`` and ``camera_id`` arguments are what make it work for any body in
:mod:`mppi.domains`.
"""

import numpy as np
from dm_control import suite

from .planner import MPPIPlanner


def run_episode(domain='walker', task='walk', *, seed=0, H=25, N=100, sigma=0.3,
                lam=0.1, gamma=1.0, planner='mppi', max_steps=1000,
                record_path=None, camera_id=0, width=640, height=480, fps=30,
                loader=None):
    """Run one MPPI-controlled episode and return its total reward.

    Args:
        domain: dm_control.suite domain name (e.g. ``'walker'``).
        task: task within the domain (e.g. ``'walk'``, ``'run'``, ``'stand'``).
        seed: RNG seed for the task and the planner (planner uses ``seed + 1``).
        H, N, sigma, lam, gamma: MPPI hyperparameters (see :class:`MPPIPlanner`).
        planner: ``'mppi'`` (softmax update) or ``'ps'`` (predictive sampling).
        max_steps: cap on environment steps.
        record_path: if set, write an mp4 of the episode to this path.
        camera_id: camera index used for rendering frames.
        width, height, fps: video parameters.
        loader: optional ``loader(task, seed) -> Environment`` for bodies without a
            built-in suite task (e.g. the Go2). When None, uses ``suite.load``.
    """
    if loader is not None:
        env = loader(task, seed)
    else:
        env = suite.load(domain, task,
                         task_kwargs={'random': np.random.RandomState(seed)})
    pl = MPPIPlanner(env, H=H, N=N, sigma=sigma, lam=lam, gamma=gamma,
                     planner=planner, seed=seed + 1)

    frames = []
    env.reset()
    if record_path:
        frames.append(env.physics.render(height=height, width=width,
                                          camera_id=camera_id))

    ret, steps = 0.0, 0
    for t in range(max_steps):
        x0 = env.physics.get_state()
        a = pl.act(x0)
        ts = env.step(a)
        ret += ts.reward
        steps += 1
        if record_path:
            frames.append(env.physics.render(height=height, width=width,
                                             camera_id=camera_id))
        if t % 50 == 0:
            print(f"step {t:4d} | return {ret:8.2f}", flush=True)
        if ts.last():
            break

    print(f"\n[{planner}] domain={domain} task={task} | return {ret:.2f} over "
          f"{steps} steps (H={H}, N={N}, sigma={sigma}, lam={lam}, gamma={gamma})")

    if record_path:
        import imageio.v2 as imageio
        with imageio.get_writer(record_path, fps=fps) as wtr:
            for f in frames:
                wtr.append_data(f)
        print(f"wrote {record_path}")
    return ret
