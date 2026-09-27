"""
Replay buffer for safe RL.
Extends QSM's ReplayBuffer to store a safety scalar alongside standard fields.
Most online tasks store signed SDF h(s); agents can derive binary or hinge
costs from h(s) when they need cumulative-budget semantics.
"""
import gym
import gym.spaces
import numpy as np

from jaxrl5.data.dataset import Dataset


def _init_replay_dict(obs_space, capacity):
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        return {k: _init_replay_dict(v, capacity) for k, v in obs_space.spaces.items()}
    else:
        raise TypeError()


def _insert_recursively(dataset_dict, data_dict, insert_index):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class SafeReplayBuffer(Dataset):
    """Replay buffer that stores (obs, action, reward, safety scalar, next_obs, mask, done)."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            costs=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )
        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self):
        return self._size

    def insert(self, data_dict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)
        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
