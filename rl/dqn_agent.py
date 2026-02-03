# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
DQN Agent for Battleship with action masking.

Implements Double DQN with experience replay and epsilon-greedy exploration.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
import copy
import logging

from rl.networks import DQNNetwork, DuelingDQNNetwork
from rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from game.battleship_game import BattleshipGame, CellState


logger = logging.getLogger(__name__)


class DQNAgent:
    """
    DQN agent for playing Battleship with action masking.
    
    Key features:
    - Action masking to prevent repeated shots
    - Double DQN to reduce overestimation
    - Experience replay for stable learning
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        board_size: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 100000,
        batch_size: int = 64,
        buffer_size: int = 50000,
        target_update_freq: int = 1000,
        use_double_dqn: bool = True,
        use_dueling: bool = False,
        use_prioritized_replay: bool = False,
    ):
        """
        Initialize DQN agent.
        
        Args:
            board_size: Size of Battleship board
            device: Device to run on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Steps to decay epsilon
            batch_size: Training batch size
            buffer_size: Replay buffer capacity
            target_update_freq: Steps between target network updates
            use_double_dqn: Use Double DQN algorithm
            use_dueling: Use Dueling DQN architecture
            use_prioritized_replay: Use prioritized experience replay
        """
        self.board_size = board_size
        self.num_actions = board_size * board_size
        self.device = torch.device(device)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        
        # Networks
        NetworkClass = DuelingDQNNetwork if use_dueling else DQNNetwork
        self.policy_net = NetworkClass(board_size=board_size).to(self.device)
        self.target_net = NetworkClass(board_size=board_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        self.use_prioritized_replay = use_prioritized_replay
        
        # Training stats
        self.steps = 0
        self.episodes = 0
        self.total_reward = 0
        
        logger.info(f"DQN Agent initialized on {device}")
        logger.info(f"Network: {NetworkClass.__name__}")
        logger.info(f"Double DQN: {use_double_dqn}")
        logger.info(f"Prioritized Replay: {use_prioritized_replay}")
    
    def state_to_tensor(self, game: BattleshipGame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert game state to neural network input.
        
        Args:
            game: BattleshipGame instance
        
        Returns:
            Tuple of (state_tensor, action_mask_tensor)
            - state_tensor: (1, 3, board_size, board_size)
            - action_mask_tensor: (1, num_actions) boolean
        """
        # Create 3-channel board representation
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        action_mask = np.zeros(self.num_actions, dtype=bool)
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                cell_state = game.player_view[row][col]
                action_idx = row * self.board_size + col
                
                if cell_state == CellState.UNEXPLORED:
                    state[0, row, col] = 1.0  # Channel 0: unexplored
                    action_mask[action_idx] = True  # Valid action
                elif cell_state == CellState.HIT:
                    state[1, row, col] = 1.0  # Channel 1: hit
                elif cell_state == CellState.MISS:
                    state[2, row, col] = 1.0  # Channel 2: miss
        
        # Convert to tensors
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        action_mask_tensor = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
        
        return state_tensor, action_mask_tensor
    
    def action_to_coordinate(self, action: int) -> str:
        """
        Convert action index to coordinate string.
        
        Args:
            action: Action index (0 to num_actions-1)
        
        Returns:
            Coordinate string (e.g., 'A5')
        """
        row = action // self.board_size
        col = action % self.board_size
        row_label = BattleshipGame.ROW_LABELS[row]
        return f"{row_label}{col + 1}"
    
    def select_action(
        self, 
        state: torch.Tensor, 
        action_mask: torch.Tensor,
        training: bool = True,
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Board state tensor
            action_mask: Valid actions mask
            training: Whether in training mode (affects epsilon)
        
        Returns:
            Selected action index
        """
        epsilon = self.epsilon if training else 0.0
        return self.policy_net.get_action(state, action_mask, epsilon)
    
    def update_epsilon(self):
        """Decay epsilon linearly."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * 
            (self.steps / self.epsilon_decay)
        )
    
    def compute_reward(
        self,
        result: Dict,
        prev_ships_sunk: int,
        current_ships_sunk: int,
        game_over: bool,
    ) -> float:
        """
        Compute reward for a transition.
        
        Reward structure:
        - Hit: +10
        - Miss: -1
        - Sink ship: +50
        - Win game: +1000
        - Move penalty: -0.1 (encourages efficiency)
        
        Args:
            result: Result dict from game.make_shot()
            prev_ships_sunk: Ships sunk before action
            current_ships_sunk: Ships sunk after action
            game_over: Whether game ended
        
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Base rewards
        if result['result'] == 'hit':
            reward += 10.0
        elif result['result'] == 'miss':
            reward -= 1.0
        
        # Ship sunk bonus
        if current_ships_sunk > prev_ships_sunk:
            reward += 50.0
        
        # Win bonus
        if game_over:
            reward += 1000.0
        
        # Efficiency penalty (encourage fewer moves)
        reward -= 0.1
        
        return reward
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(
            state, action, reward, next_state, done,
            action_mask, next_action_mask
        )
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch
        if self.use_prioritized_replay:
            (states, actions, rewards, next_states, dones,
             action_masks, next_action_masks, indices, weights) = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.from_numpy(weights).float().to(self.device)
        else:
            (states, actions, rewards, next_states, dones,
             action_masks, next_action_masks) = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        # Convert to tensors
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        next_action_masks = torch.from_numpy(next_action_masks).bool().to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use policy net to select action, target net to evaluate
                next_q_policy = self.policy_net(next_states)
                next_q_policy[~next_action_masks] = float('-inf')
                next_actions = next_q_policy.argmax(dim=1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_net(next_states)
                next_q_values[~next_action_masks] = float('-inf')
                next_q_values = next_q_values.max(dim=1)[0]
            
            # Handle terminal states
            next_q_values[dones == 1] = 0.0
            
            # Compute target Q values
            target_q_values = rewards + self.gamma * next_q_values
        
        # Compute loss
        td_errors = target_q_values - current_q_values
        loss = (td_errors.pow(2) * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay:
            priorities = td_errors.abs().detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, priorities)
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def play_episode(
        self,
        game: Optional[BattleshipGame] = None,
        max_steps: int = 100,
        training: bool = True,
        verbose: bool = False,
    ) -> Dict:
        """
        Play one episode of Battleship.
        
        Args:
            game: BattleshipGame instance (creates new if None)
            max_steps: Maximum steps per episode
            training: Whether to train during episode
            verbose: Print debug information
        
        Returns:
            Episode statistics dict
        """
        if game is None:
            game = BattleshipGame()
        
        episode_reward = 0
        episode_loss = []
        moves = 0
        
        for step in range(max_steps):
            # Get current state
            state, action_mask = self.state_to_tensor(game)
            prev_ships_sunk = game.get_game_status()['ships_sunk']
            
            # Select action
            action = self.select_action(state, action_mask, training=training)
            coordinate = self.action_to_coordinate(action)
            
            # Take action
            result = game.make_shot(coordinate)
            
            if not result['valid']:
                logger.error(f"Invalid action selected: {coordinate}")
                break
            
            moves += 1
            
            # Get next state
            next_state, next_action_mask = self.state_to_tensor(game)
            current_ships_sunk = game.get_game_status()['ships_sunk']
            game_over = game.is_game_over()
            
            # Compute reward
            reward = self.compute_reward(result, prev_ships_sunk, current_ships_sunk, game_over)
            episode_reward += reward
            
            if training:
                # Store transition
                self.store_transition(
                    state.cpu().numpy()[0],
                    action,
                    reward,
                    next_state.cpu().numpy()[0],
                    game_over,
                    action_mask.cpu().numpy()[0],
                    next_action_mask.cpu().numpy()[0],
                )
                
                # Train
                loss = self.train_step()
                if loss is not None:
                    episode_loss.append(loss)
                
                # Update target network
                if self.steps % self.target_update_freq == 0:
                    self.update_target_network()
                
                # Update epsilon
                self.update_epsilon()
                self.steps += 1
            
            if verbose:
                print(f"Step {step+1}: {coordinate} -> {result['message']}, Reward: {reward:.1f}")
            
            # Check if game over
            if game_over:
                break
        
        if training:
            self.episodes += 1
        
        # Collect statistics
        stats = game.get_game_status()
        return {
            'reward': episode_reward,
            'loss': np.mean(episode_loss) if episode_loss else 0.0,
            'moves': moves,
            'won': stats['game_over'],
            'ships_sunk': stats['ships_sunk'],
            'hits': stats['hits'],
            'misses': stats['misses'],
            'accuracy': stats['hits'] / stats['shots_fired'] if stats['shots_fired'] > 0 else 0,
            'epsilon': self.epsilon,
        }
    
    def save(self, filepath: str):
        """Save agent state to file."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
        }, filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.epsilon = checkpoint['epsilon']
        logger.info(f"Agent loaded from {filepath}")
