"""Receding-horizon MPPI planner over the true MuJoCo dynamics.

This module is **body-agnostic**: the planner reads everything it needs (action
dimension, bounds, control/physics timesteps, reward) from the DM-Control
environment it is handed, so the same code drives the walker, the Unitree Go2, or
any other body. See :mod:`mppi.domains` for the per-body
configuration and :func:`mppi.runner.run_episode` for the episode loop.
"""

import numpy as np


class MPPIPlanner:
    """Receding-horizon MPPI over the true simulator dynamics.

    Uses the *true* MuJoCo simulator as the model and the *exact* DM-Control
    reward as the objective, so there are no dynamics derivatives, no surrogate
    cost, and no training. Each control step:

      1. snapshot the real sim state ``x0``,
      2. sample ``N`` action sequences by perturbing a nominal mean with Gaussian
         noise,
      3. roll each candidate through a planning copy of the sim, summing the true
         (optionally discounted) reward over the horizon,
      4. reward-weight the candidates (softmax) to update the nominal,
      5. execute the nominal's first action and shift the nominal forward.

    The planning sim is a copy of the real env's physics (``share_model=True``),
    so candidate rollouts never disturb the real episode. Rewards come straight
    from the task, so the planner optimizes the exact DM objective.

    Passing ``planner='ps'`` swaps the softmax update for MJPC "Predictive
    Sampling" (a hard argmax over candidates).

    The nominal is warm-started and refilled with a *rest action* ``u_ref``. For
    motor (torque) bodies like the walker that is zero. For a
    position-controlled body like the Go2, ``u_ref`` is the standing-pose joint
    targets, so the planner samples small target offsets around the stance rather
    than around an all-zero (collapsed) pose. ``u_ref`` defaults to the task's
    ``nominal_action`` attribute when present, else zeros.
    """

    def __init__(self, env, H=25, N=100, sigma=0.3, lam=0.1, gamma=1.0,
                 planner='mppi', seed=0, u_ref=None):
        self.task = env.task
        self.plan_phys = env.physics.copy(share_model=True)
        # control step = control_timestep / physics_timestep physics substeps
        self.nsub = int(round(env.control_timestep() / env.physics.timestep()))

        spec = env.action_spec()
        self.udim = spec.shape[0]
        self.lo = spec.minimum.astype(np.float64)
        self.hi = spec.maximum.astype(np.float64)

        self.H, self.N = H, N
        self.sigma, self.lam, self.gamma = sigma, lam, gamma
        self.planner = planner
        self.rng = np.random.RandomState(seed)

        # Rest action the nominal warm-starts / refills with (0 for torque bodies,
        # the standing pose for a position-controlled body like the Go2).
        if u_ref is None:
            u_ref = getattr(env.task, 'nominal_action', None)
        self.u_ref = (np.zeros(self.udim) if u_ref is None
                      else np.asarray(u_ref, dtype=np.float64))
        self.mu = np.tile(self.u_ref, (H, 1))  # nominal mean

    def rollout_return(self, x0, U):
        """Sum of (discounted) true reward from x0 under action sequence U."""
        p = self.plan_phys
        with p.reset_context():
            p.set_state(x0)
        R, disc = 0.0, 1.0
        for h in range(self.H):
            p.set_control(np.clip(U[h], self.lo, self.hi))
            for _ in range(self.nsub):
                p.step()
            R += disc * self.task.get_reward(p)
            disc *= self.gamma
        return R

    def act(self, x0):
        """Run one MPPI step from real state x0; return the action to execute."""
        # Sample candidates: N-1 noisy perturbations of the nominal + the nominal
        # itself (so the update can never select something worse than warm-start).
        eps = self.rng.randn(self.N - 1, self.H, self.udim) * self.sigma
        cand = np.clip(self.mu[None] + eps, self.lo, self.hi)
        cand = np.concatenate([self.mu[None], cand], axis=0)  # (N, H, udim)

        R = np.array([self.rollout_return(x0, cand[n]) for n in range(self.N)])

        if self.planner == 'ps':
            # MJPC Predictive Sampling: keep the single best candidate.
            self.mu = cand[int(np.argmax(R))].copy()
        else:
            # MPPI: reward-weighted (softmax) average of the candidates.
            w = np.exp((R - R.max()) / self.lam)
            w /= w.sum()
            self.mu = np.tensordot(w, cand, axes=(0, 0))

        a = self.mu[0].copy()
        # Shift nominal forward for warm-starting the next control step.
        self.mu[:-1] = self.mu[1:]
        self.mu[-1] = self.u_ref
        return np.clip(a, self.lo, self.hi)
