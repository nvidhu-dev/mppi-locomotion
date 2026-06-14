"""Per-body configuration registry — the single place to switch the robot.

Each body the planner can drive is one :class:`DomainConfig` entry in
:data:`DOMAINS`. The config bundles the ``dm_control.suite`` domain name, the
tasks you can select, the render camera, and starting-point MPPI hyperparameters
tuned for that body. The CLI uses these values as *defaults*, so every field
stays overridable on the command line.

Most bodies come straight from ``dm_control.suite`` (set ``domain`` to the suite
name). A body with no built-in suite task — like the Unitree Go2 — instead
provides a ``loader`` callable that builds its own ``control.Environment``.

Adding a new suite body is a one-entry change here — no other module needs to
know about it. For example::

    DOMAINS['cheetah'] = DomainConfig(
        'cheetah', tasks=('run',), camera_id=0, H=25, N=100, sigma=0.3,
    )
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class DomainConfig:
    """MPPI defaults and metadata for one body."""

    domain: str          # dm_control.suite domain name (or a label, if loader is set)
    tasks: tuple         # selectable task names; tasks[0] is the default task
    camera_id: int = 0   # camera index passed to physics.render()
    # MPPI hyperparameters (all overridable on the CLI)
    H: int = 25          # planning horizon (control steps)
    N: int = 100         # number of sampled action sequences per step
    sigma: float = 0.3   # std of the Gaussian action noise
    lam: float = 0.1     # softmax temperature for the MPPI reward weighting
    gamma: float = 1.0   # reward discount over the horizon
    # Optional custom env factory: loader(task, seed) -> dm_control Environment.
    # When None, the runner uses suite.load(domain, task, ...).
    loader: Optional[Callable] = None


def _load_go2(task, seed):
    """Lazy Go2 env factory (imports dm_control/robot_descriptions on first use)."""
    from .go2 import make_env
    return make_env(task=task, seed=seed)


# The walker entry reproduces the original mppi_walker.py defaults exactly.
DOMAINS = {
    'walker': DomainConfig(
        domain='walker', tasks=('walk', 'stand', 'run'),
        camera_id=0, H=25, N=100, sigma=0.3, lam=0.1, gamma=1.0,
    ),
    # Unitree Go2: a real robot model (MuJoCo Menagerie) with no built-in task,
    # so it supplies its own env via mppi.go2.make_env. Position (PD) actuators,
    # so the action is a joint-target offset in radians and sigma is small. Camera
    # 0 is the chase camera added in mppi/go2.py.
    'go2': DomainConfig(
        domain='go2', tasks=('walk', 'stand', 'run'),
        camera_id=0, H=16, N=150, sigma=0.3, lam=0.05, gamma=1.0,
        loader=_load_go2,
    ),
}


def get_config(domain):
    """Return the :class:`DomainConfig` for ``domain`` or raise a clear error."""
    try:
        return DOMAINS[domain]
    except KeyError:
        valid = ', '.join(sorted(DOMAINS))
        raise ValueError(
            f"unknown domain {domain!r}; valid domains are: {valid}"
        ) from None
