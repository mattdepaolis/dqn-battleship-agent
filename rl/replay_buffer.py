# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Experience replay buffers for DQN training.

Provides standard and prioritized experience replay.
"""

import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple


# Experience tuple
Experience = namedtuple(
    'Experience',
    ['state', 'action', 'reward', 'next_state', 'done', 'action_mask', 'next_action_mask']
)


class ReplayBuffer:
    """
    Standard experience replay buffer for DQN.
    
    Stores transitions and samples uniformly for training.
    """
    
    def __init__(self, capacity: int = 50000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current board state (3, board_size, board_size)
            action: Action taken (0 to num_actions-1)
            reward: Reward received
            next_state: Next board state
            done: Whether episode ended
            action_mask: Valid actions in current state
            next_action_mask: Valid actions in next state
        """
        experience = Experience(
            state, action, reward, next_state, done,
            action_mask, next_action_mask
        )
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of experiences uniformly.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, 
                     action_masks, next_action_masks)
        """
        experiences = random.sample(self.buffer, batch_size)
        
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        action_masks = np.array([e.action_mask for e in experiences])
        next_action_masks = np.array([e.next_action_mask for e in experiences])
        
        return (
            states, actions, rewards, next_states, dones,
            action_masks, next_action_masks
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    
    Samples experiences based on TD error for more efficient learning.
    Prioritizes rare/important events (e.g., sinking ships).
    """
    
    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ):
        """Add experience with maximum priority."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        experience = Experience(
            state, action, reward, next_state, done,
            action_mask, next_action_mask
        )
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch based on priorities.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones,
                     action_masks, next_action_masks, indices, weights)
        """
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        beta = self._get_beta()
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        action_masks = np.array([e.action_mask for e in experiences])
        next_action_masks = np.array([e.next_action_mask for e in experiences])
        
        return (
            states, actions, rewards, next_states, dones,
            action_masks, next_action_masks, indices, weights
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Indices of experiences to update
            td_errors: TD errors for priority calculation
        """
        for idx, td_error in zip(indices, td_errors):
            # Add small constant to avoid zero priority
            self.priorities[idx] = abs(td_error) + 1e-6
    
    def _get_beta(self) -> float:
        """Calculate current beta value (annealed over time)."""
        beta = self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames
        self.frame += 1
        return min(1.0, beta)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size


class SumTree:
    """
    Sum tree data structure for efficient prioritized sampling.
    
    Alternative implementation for large-scale training.
    """
    
    def __init__(self, capacity: int):
        """Initialize sum tree with given capacity."""
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve sample index based on priority sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Return sum of all priorities."""
        return self.tree[0]
    
    def add(self, priority: float, data: Experience):
        """Add new experience with priority."""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """Update priority for given index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Experience]:
        """Get experience based on priority sampling."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
