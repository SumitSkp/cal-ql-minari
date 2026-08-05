"""Small smoke tests for the Minari + safety-ablation merge.

This is not a full experiment. It just checks the parts we touched:
ReplayBuffer cost fields, Gymnasium-style sampling, the normal SAC/CQL update,
and one tiny SAC update with the optional Q_fall critic switched on.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from JaxCQL.conservative_sac import ConservativeSAC
from JaxCQL.model import FullyConnectedQFunction, TanhGaussianPolicy
from JaxCQL.replay_buffer import ReplayBuffer
from JaxCQL.sampler import TrajSampler
from JaxCQL.utils import set_random_seed


class _Space:
    def __init__(self, shape):
        self.shape = shape


class _Spec:
    id = "DummySafety-v0"
    name = "DummySafety-v0"
    max_episode_steps = 3


class _TerminatingEnv:
    spec = _Spec()
    observation_space = _Space((2,))
    action_space = _Space((1,))

    def __init__(self, truncate=False):
        self._step = 0
        self._truncate = truncate

    def reset(self):
        self._step = 0
        return np.array([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.array([float(self._step), 0.0], dtype=np.float32)
        reward = 1.0
        terminated = self._step == 2 and not self._truncate
        truncated = self._step == 1 and self._truncate
        return obs, reward, terminated, truncated, {}


def _constant_policy(observations, deterministic=False):
    return np.zeros((observations.shape[0], 1), dtype=np.float32)


def check_replay_buffer():
    batch = dict(
        observations=np.zeros((3, 2), dtype=np.float32),
        actions=np.zeros((3, 1), dtype=np.float32),
        rewards=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        next_observations=np.ones((3, 2), dtype=np.float32),
        dones=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        costs=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        truncations=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        mc_returns=np.array([5.0, 4.0, 3.0], dtype=np.float32),
    )
    buffer = ReplayBuffer(10, data=batch)
    stored = buffer.data
    assert stored["costs"].tolist() == [0.0, 0.0, 1.0]
    assert stored["truncations"].tolist() == [0.0, 1.0, 0.0]
    print("ok: replay buffer keeps costs and truncations")


def check_sampler_costs():
    standard_sampler = TrajSampler(_TerminatingEnv())
    standard_traj = standard_sampler.sample(_constant_policy, 1)[0]
    assert float(np.sum(standard_traj["raw_rewards"])) == 2.0
    assert float(np.sum(standard_traj["rewards"])) == 2.0
    assert float(np.sum(standard_traj["costs"])) == 1.0

    sampler = TrajSampler(
        _TerminatingEnv(),
        use_mc=True,
        use_fall_penalty=True,
        fall_penalty=100.0,
    )
    traj = sampler.sample(_constant_policy, 1)[0]
    assert float(np.sum(traj["raw_rewards"])) == 2.0
    assert float(np.sum(traj["costs"])) == 1.0
    assert float(np.sum(traj["rewards"])) == -98.0

    truncated_sampler = TrajSampler(_TerminatingEnv(truncate=True))
    truncated_traj = truncated_sampler.sample(_constant_policy, 1)[0]
    assert float(np.sum(truncated_traj["rewards"])) == 1.0
    assert float(np.sum(truncated_traj["costs"])) == 0.0
    assert float(np.sum(truncated_traj["truncations"])) == 1.0
    print("ok: sampler separates fall terminations from time truncations")


def check_one_cost_critic_update():
    set_random_seed(0)
    observation_dim = 3
    action_dim = 2
    config = ConservativeSAC.get_default_config(
        dict(
            use_cost_critic=True,
            use_automatic_entropy_tuning=False,
            backup_entropy=False,
            cql_n_actions=2,
            cql_max_target_backup=False,
            policy_lr=1e-4,
            qf_lr=1e-4,
            cost_qf_lr=1e-4,
        )
    )
    policy = TanhGaussianPolicy(observation_dim, action_dim, arch="16-16")
    qf = FullyConnectedQFunction(observation_dim, action_dim, arch="16-16")
    sac = ConservativeSAC(config, policy, qf)

    batch_size = 6
    batch = dict(
        observations=np.zeros((batch_size, observation_dim), dtype=np.float32),
        actions=np.zeros((batch_size, action_dim), dtype=np.float32),
        rewards=np.ones(batch_size, dtype=np.float32),
        next_observations=np.ones((batch_size, observation_dim), dtype=np.float32),
        dones=np.zeros(batch_size, dtype=np.float32),
        costs=np.array([0, 0, 1, 0, 0, 1], dtype=np.float32),
        mc_returns=np.ones(batch_size, dtype=np.float32),
    )
    metrics = sac.train(
        batch,
        use_cql=True,
        cql_min_q_weight=0.1,
        enable_calql=True,
        cost_lambda=10.0,
    )
    assert metrics["use_cost_critic"] == 1
    assert "cost_qf1_loss" in metrics
    assert "cql/cql_q1_next_actions_mean" in metrics
    for key in ("policy_loss", "cost_qf1_loss", "cost_qf2_loss"):
        assert np.all(np.isfinite(np.asarray(metrics[key])))
    print("ok: one SAC update runs with Q_fall cost critic")


def check_one_standard_update():
    set_random_seed(1)
    observation_dim = 3
    action_dim = 2
    config = ConservativeSAC.get_default_config(
        dict(
            use_cost_critic=False,
            use_automatic_entropy_tuning=False,
            backup_entropy=False,
            cql_n_actions=2,
            cql_max_target_backup=False,
            policy_lr=1e-4,
            qf_lr=1e-4,
        )
    )
    policy = TanhGaussianPolicy(observation_dim, action_dim, arch="16-16")
    qf = FullyConnectedQFunction(observation_dim, action_dim, arch="16-16")
    sac = ConservativeSAC(config, policy, qf)

    batch_size = 6
    batch = dict(
        observations=np.zeros((batch_size, observation_dim), dtype=np.float32),
        actions=np.zeros((batch_size, action_dim), dtype=np.float32),
        rewards=np.ones(batch_size, dtype=np.float32),
        next_observations=np.ones((batch_size, observation_dim), dtype=np.float32),
        dones=np.zeros(batch_size, dtype=np.float32),
        mc_returns=np.ones(batch_size, dtype=np.float32),
    )
    metrics = sac.train(
        batch,
        use_cql=True,
        cql_min_q_weight=0.1,
        enable_calql=True,
    )
    assert metrics["use_cost_critic"] == 0
    assert "qf1_loss" in metrics
    assert "cql/cql_q1_next_actions_mean" in metrics
    for key in ("policy_loss", "qf1_loss", "qf2_loss"):
        assert np.all(np.isfinite(np.asarray(metrics[key])))
    print("ok: normal Cal-QL/CQL update still runs")


if __name__ == "__main__":
    check_replay_buffer()
    check_sampler_costs()
    check_one_standard_update()
    check_one_cost_critic_update()
    print("all safety smoke tests passed")
