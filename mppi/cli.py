"""Command-line interface.

Resolves ``--domain`` first, pulls that body's defaults from :mod:`mppi.domains`,
and uses them as the defaults for every MPPI / task / camera flag — so each body
runs well out of the box while everything stays overridable.
"""

import argparse

from .domains import DOMAINS, get_config


def build_parser():
    p = argparse.ArgumentParser(
        description="MPPI / Predictive-Sampling MPC on the true DM-Control sim.")
    p.add_argument('--domain', type=str, default='walker',
                   choices=sorted(DOMAINS),
                   help="body to control (sets per-body defaults below)")
    # Defaults of None mean "fill from the domain config after --domain is known".
    p.add_argument('--task', type=str, default=None,
                   help="task within the domain (default: the domain's first task)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--horizon', type=int, default=None, help="planning horizon H")
    p.add_argument('--samples', type=int, default=None, help="rollouts per step N")
    p.add_argument('--sigma', type=float, default=None, help="action noise std")
    p.add_argument('--lam', type=float, default=None, help="softmax temperature")
    p.add_argument('--gamma', type=float, default=None, help="reward discount")
    p.add_argument('--planner', type=str, default='mppi', choices=['mppi', 'ps'])
    p.add_argument('--max_steps', type=int, default=1000)
    p.add_argument('--camera', type=int, default=None, help="render camera id")
    p.add_argument('--record', type=str, default=None, help="output mp4 path")
    p.add_argument('--record_width', type=int, default=640)
    p.add_argument('--record_height', type=int, default=480)
    p.add_argument('--record_fps', type=int, default=30)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = get_config(args.domain)

    # Fall back to the domain's defaults for anything left unset on the CLI.
    task = args.task if args.task is not None else cfg.tasks[0]
    if task not in cfg.tasks:
        valid = ', '.join(cfg.tasks)
        raise SystemExit(
            f"task {task!r} is not valid for domain {args.domain!r}; "
            f"choose one of: {valid}")

    H = args.horizon if args.horizon is not None else cfg.H
    N = args.samples if args.samples is not None else cfg.N
    sigma = args.sigma if args.sigma is not None else cfg.sigma
    lam = args.lam if args.lam is not None else cfg.lam
    gamma = args.gamma if args.gamma is not None else cfg.gamma
    camera_id = args.camera if args.camera is not None else cfg.camera_id

    from .runner import run_episode  # deferred: needs dm_control / mujoco
    run_episode(domain=cfg.domain, task=task, seed=args.seed, H=H, N=N,
                sigma=sigma, lam=lam, gamma=gamma, planner=args.planner,
                max_steps=args.max_steps, record_path=args.record,
                camera_id=camera_id, width=args.record_width,
                height=args.record_height, fps=args.record_fps,
                loader=cfg.loader)


if __name__ == '__main__':
    main()
