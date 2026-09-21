"""NumPy prioritized n-step replay with independent queues for vectorized games.

Rewards passed to add() must already have their intended scaling/shaping. True
termination suppresses bootstrapping; a time limit flushes its queue but retains
bootstrapping. Observations and the actual final next-observations are stored in
full, so frame stacks remain correct for multi-step transitions.

state_dict() includes NumPy arrays and RNG state. When using torch serialization,
load these snapshots only from trusted local files with weights_only=False.
"""

from collections import deque
from copy import deepcopy

import numpy as np


class PrioritizedNStepReplay:
    def __init__(self, capacity, n_step=3, gamma=0.99, alpha=0.5, seed=43,
                 n_envs=4, observation_shape=(4, 84, 84)):
        for name, value in (("capacity", capacity), ("n_step", n_step), ("n_envs", n_envs)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite and between zero and one")
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and between zero and one")
        if not observation_shape or any(not isinstance(x, (int, np.integer)) or x < 1
                                        for x in observation_shape):
            raise ValueError("observation_shape must contain positive integer dimensions")
        self.capacity = int(capacity)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.n_envs = int(n_envs)
        self.observation_shape = tuple(int(x) for x in observation_shape)
        self.epsilon = 1e-6
        self.rng = np.random.default_rng(seed)
        self.obs = np.empty((self.capacity, *self.observation_shape), dtype=np.uint8)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.discounts = np.empty(self.capacity, dtype=np.float32)
        self.horizons = np.empty(self.capacity, dtype=np.int32)
        self.priorities = np.empty(self.capacity, dtype=np.float64)
        self._leaf_count = 1 << (self.capacity - 1).bit_length()
        self._sum_tree = np.zeros(2 * self._leaf_count, dtype=np.float64)
        self._min_tree = np.full(2 * self._leaf_count, np.inf, dtype=np.float64)
        self.pending = [deque() for _ in range(self.n_envs)]
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def __len__(self):
        return self.size

    def _observation(self, value):
        array = np.asarray(value)
        if array.shape != self.observation_shape or array.dtype != np.uint8:
            raise ValueError(f"Expected uint8 observation shape {self.observation_shape}")
        return array.copy()

    def _transition(self, obs, action, reward, next_obs, terminated, truncated):
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer")
        reward = float(reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        return (self._observation(obs), int(action), reward,
                self._observation(next_obs), bool(terminated), bool(truncated))

    def add(self, env_id, obs, action, reward, next_obs, terminated, truncated):
        if not isinstance(env_id, (int, np.integer)) or not 0 <= env_id < self.n_envs:
            raise ValueError("env_id is outside the configured environment range")
        transition = self._transition(obs, action, reward, next_obs, terminated, truncated)
        queue = self.pending[env_id]
        queue.append(transition)
        if len(queue) >= self.n_step:
            self._emit(queue)
            queue.popleft()
        if terminated or truncated:
            while queue:
                self._emit(queue)
                queue.popleft()

    def _emit(self, queue):
        total = 0.0
        factor = 1.0
        horizon = 0
        for transition in queue:
            total += factor * transition[2]
            horizon += 1
            factor *= self.gamma
            if transition[4] or transition[5] or horizon == self.n_step:
                break
        if not np.isfinite(total) or abs(total) > np.finfo(np.float32).max:
            raise ValueError("n-step return cannot be represented as a finite float32")
        index = self.position
        self.obs[index] = queue[0][0]
        self.actions[index] = queue[0][1]
        self.rewards[index] = total
        self.next_obs[index] = transition[3]
        self.discounts[index] = 0.0 if transition[4] else factor
        self.horizons[index] = horizon
        self.priorities[index] = self.max_priority
        self._set_leaves(np.array([index]), np.array([self.max_priority ** self.alpha]))
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _set_leaves(self, indices, masses):
        nodes = np.asarray(indices, dtype=np.int64) + self._leaf_count
        self._sum_tree[nodes] = masses
        self._min_tree[nodes] = masses
        nodes = np.unique(nodes // 2)
        while nodes.size and nodes[0] > 0:
            self._sum_tree[nodes] = self._sum_tree[2 * nodes] + self._sum_tree[2 * nodes + 1]
            self._min_tree[nodes] = np.minimum(self._min_tree[2 * nodes],
                                              self._min_tree[2 * nodes + 1])
            nodes = np.unique(nodes // 2)

    def sample(self, batch_size, beta):
        if self.size == 0:
            raise ValueError("Cannot sample an empty replay")
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("beta must be finite and between zero and one")
        total = self._sum_tree[1]
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Invalid total replay priority")
        # One independently jittered sample from each equal-mass stratum.
        values = (np.arange(batch_size) + self.rng.random(batch_size)) * (total / batch_size)
        values = np.minimum(values, np.nextafter(total, 0.0))
        nodes = np.ones(batch_size, dtype=np.int64)
        for _ in range(self._leaf_count.bit_length() - 1):
            left = 2 * nodes
            left_mass = self._sum_tree[left]
            take_right = values >= left_mass
            values -= np.where(take_right, left_mass, 0.0)
            nodes = left + take_right
        indices = nodes - self._leaf_count
        if np.any(indices >= self.size):
            raise RuntimeError("Priority tree selected an unfilled replay slot")
        # Globally normalized (N*p)^-beta, using the minimum active probability.
        # N and total cancel in the ratio; log space avoids intermediate overflow.
        weights = np.exp(-float(beta) * (np.log(self._sum_tree[nodes]) -
                                         np.log(self._min_tree[1]))).astype(np.float32)
        return {"obs": self.obs[indices], "actions": self.actions[indices],
                "rewards": self.rewards[indices], "next_obs": self.next_obs[indices],
                "discounts": self.discounts[indices], "weights": weights,
                "indices": indices}

    def update_priorities(self, indices, td_errors):
        indices = np.asarray(indices)
        errors = np.asarray(td_errors, dtype=np.float64)
        if indices.ndim != 1 or errors.ndim != 1 or indices.shape != errors.shape:
            raise ValueError("indices and td_errors must be equal-length one-dimensional arrays")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("indices must be integers")
        if not indices.size:
            return
        if np.any(indices < 0) or np.any(indices >= self.size) or not np.all(np.isfinite(errors)):
            raise ValueError("Invalid replay indices or non-finite TD errors")
        unique, inverse = np.unique(indices, return_inverse=True)
        priorities = np.zeros(len(unique), dtype=np.float64)
        np.maximum.at(priorities, inverse, np.abs(errors) + self.epsilon)
        masses = priorities ** self.alpha
        other_total = self._sum_tree[1] - self._sum_tree[unique + self._leaf_count].sum()
        if not np.all(np.isfinite(masses)) or not np.isfinite(other_total + masses.sum()):
            raise ValueError("Updated priorities would overflow the priority tree")
        self.priorities[unique] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self._set_leaves(unique, masses)

    def state_dict(self, *, copy_arrays=True):
        """Return a complete snapshot, including pending tails and sampler RNG.

        copy_arrays=False avoids duplicating replay storage, but its returned
        array views must be serialized synchronously before collecting more data.
        Uninitialized storage beyond size is never exported.
        """
        arrays = {}
        for name in ("obs", "next_obs", "actions", "rewards", "discounts", "horizons", "priorities"):
            value = getattr(self, name)[:self.size]
            arrays[name] = value.copy() if copy_arrays else value
        return {"schema_version": 1,
                "config": {"capacity": self.capacity, "n_step": self.n_step,
                           "gamma": self.gamma, "alpha": self.alpha,
                           "n_envs": self.n_envs, "observation_shape": self.observation_shape,
                           "epsilon": self.epsilon},
                "size": self.size, "position": self.position,
                "max_priority": self.max_priority, "arrays": arrays,
                "pending": deepcopy([list(queue) for queue in self.pending]),
                "rng_state": deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        expected = {"capacity": self.capacity, "n_step": self.n_step,
                    "gamma": self.gamma, "alpha": self.alpha,
                    "n_envs": self.n_envs, "observation_shape": self.observation_shape,
                    "epsilon": self.epsilon}
        if state.get("schema_version") != 1 or state.get("config") != expected:
            raise ValueError("Replay snapshot configuration does not match this replay")
        size, position = int(state["size"]), int(state["position"])
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("Invalid replay snapshot size or position")
        if size < self.capacity and position != size:
            raise ValueError("Partially filled replay snapshot has an invalid position")
        arrays = {}
        for name in ("obs", "next_obs", "actions", "rewards", "discounts", "horizons", "priorities"):
            value = np.asarray(state["arrays"][name])
            destination = getattr(self, name)
            if value.shape != (size, *destination.shape[1:]) or value.dtype != destination.dtype:
                raise ValueError(f"Invalid snapshot array {name}")
            arrays[name] = value
        if not all(np.all(np.isfinite(arrays[name])) for name in ("rewards", "discounts", "priorities")):
            raise ValueError("Non-finite replay snapshot values")
        if np.any(arrays["priorities"] <= 0) or np.any(arrays["horizons"] < 1) or np.any(arrays["horizons"] > self.n_step):
            raise ValueError("Invalid priorities or horizons in replay snapshot")
        if np.any(arrays["discounts"] < 0) or np.any(arrays["discounts"] > 1):
            raise ValueError("Invalid replay snapshot discounts")
        maximum = float(state["max_priority"])
        if not np.isfinite(maximum) or maximum <= 0 or (size and maximum < arrays["priorities"].max()):
            raise ValueError("Invalid maximum priority in replay snapshot")
        if len(state["pending"]) != self.n_envs:
            raise ValueError("Wrong number of pending environment queues")
        pending = []
        for queue in state["pending"]:
            if len(queue) >= self.n_step:
                raise ValueError("Unflushed n-step queue in replay snapshot")
            converted = deque(self._transition(*transition) for transition in queue)
            if any(transition[4] or transition[5] for transition in converted):
                raise ValueError("Pending snapshot queues must not cross episode boundaries")
            pending.append(converted)
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state["rng_state"])
        masses = arrays["priorities"] ** self.alpha
        if not np.isfinite(masses.sum()):
            raise ValueError("Replay snapshot priorities overflow")
        for name, value in arrays.items():
            getattr(self, name)[:size] = value
        self.size, self.position, self.max_priority = size, position, maximum
        self.pending, self.rng = pending, rng
        self._sum_tree.fill(0.0)
        self._min_tree.fill(np.inf)
        self._sum_tree[self._leaf_count:self._leaf_count + size] = masses
        self._min_tree[self._leaf_count:self._leaf_count + size] = masses
        start = self._leaf_count // 2
        while start:
            nodes = np.arange(start, 2 * start)
            self._sum_tree[nodes] = self._sum_tree[2 * nodes] + self._sum_tree[2 * nodes + 1]
            self._min_tree[nodes] = np.minimum(self._min_tree[2 * nodes], self._min_tree[2 * nodes + 1])
            start //= 2
