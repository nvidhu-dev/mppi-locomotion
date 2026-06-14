"""Sampling-based MPC (MPPI / Predictive Sampling) on the true DM-Control sim.

``run_episode`` is imported lazily so that the lightweight pieces (``domains``,
``planner``) stay importable even where ``dm_control`` is not installed.
"""

from .domains import DOMAINS, DomainConfig, get_config
from .planner import MPPIPlanner

__all__ = [
    'DOMAINS', 'DomainConfig', 'get_config', 'MPPIPlanner', 'run_episode',
]


def __getattr__(name):
    if name == 'run_episode':
        from .runner import run_episode
        return run_episode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
