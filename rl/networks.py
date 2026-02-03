# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Neural network architectures for DQN Battleship agent.

Provides CNN-based Q-network with action masking support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Battleship with action masking.
    
    Architecture:
    - Input: 10x10x3 board state (unexplored, hit, miss channels)
    - 3 Conv2D layers with batch normalization
    - Fully connected layers
    - Output: Q-values for all 100 actions
    - Action masking applied before selecting action
    """
    
    def __init__(
        self,
        board_size: int = 10,
        num_channels: int = 3,
        conv_channels: Tuple[int, int, int] = (32, 64, 128),
        fc_hidden_size: int = 512,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize DQN network.
        
        Args:
            board_size: Size of the board (default 10 for 10x10)
            num_channels: Number of input channels (3: unexplored, hit, miss)
            conv_channels: Tuple of channel sizes for conv layers
            fc_hidden_size: Size of fully connected hidden layer
            dropout_rate: Dropout rate for regularization
        """
        super(DQNNetwork, self).__init__()
        
        self.board_size = board_size
        self.num_actions = board_size * board_size
        
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(
            num_channels, 
            conv_channels[0], 
            kernel_size=3, 
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        
        self.conv2 = nn.Conv2d(
            conv_channels[0], 
            conv_channels[1], 
            kernel_size=3, 
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(conv_channels[1])
        
        self.conv3 = nn.Conv2d(
            conv_channels[1], 
            conv_channels[2], 
            kernel_size=3, 
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(conv_channels[2])
        
        # Calculate flattened size after convolutions
        # With padding=1, spatial dimensions stay same
        conv_output_size = conv_channels[2] * board_size * board_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, fc_hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc_hidden_size, self.num_actions)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, board_size, board_size)
               Channel 0: Unexplored cells (1 where unexplored, 0 otherwise)
               Channel 1: Hit cells (1 where hit, 0 otherwise)
               Channel 2: Miss cells (1 where miss, 0 otherwise)
        
        Returns:
            Q-values tensor of shape (batch_size, num_actions)
        """
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        q_values = self.fc2(x)
        
        return q_values
    
    def get_action(
        self, 
        state: torch.Tensor, 
        action_mask: torch.Tensor,
        epsilon: float = 0.0,
    ) -> int:
        """
        Select action using epsilon-greedy policy with action masking.
        
        Args:
            state: Board state tensor (1, 3, board_size, board_size)
            action_mask: Boolean mask (1, num_actions) where True = valid action
            epsilon: Exploration rate (0 = greedy, 1 = random)
        
        Returns:
            Selected action index (0 to num_actions-1)
        """
        # Epsilon-greedy exploration
        if torch.rand(1).item() < epsilon:
            # Random action from valid actions
            valid_actions = torch.where(action_mask[0])[0]
            if len(valid_actions) == 0:
                raise ValueError("No valid actions available!")
            return valid_actions[torch.randint(len(valid_actions), (1,))].item()
        
        # Greedy action selection with masking
        with torch.no_grad():
            q_values = self.forward(state)
            
            # Apply action mask: set invalid actions to -inf
            masked_q_values = q_values.clone()
            masked_q_values[~action_mask] = float('-inf')
            
            # Select action with highest Q-value
            action = masked_q_values.argmax(dim=1).item()
            
            return action


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture for Battleship.
    
    Separates value and advantage streams for better learning.
    Can be used as an alternative to standard DQN.
    """
    
    def __init__(
        self,
        board_size: int = 10,
        num_channels: int = 3,
        conv_channels: Tuple[int, int, int] = (32, 64, 128),
        fc_hidden_size: int = 512,
        dropout_rate: float = 0.3,
    ):
        """Initialize Dueling DQN network."""
        super(DuelingDQNNetwork, self).__init__()
        
        self.board_size = board_size
        self.num_actions = board_size * board_size
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(num_channels, conv_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])
        
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_channels[2])
        
        conv_output_size = conv_channels[2] * board_size * board_size
        
        # Value stream
        self.value_fc1 = nn.Linear(conv_output_size, fc_hidden_size)
        self.value_fc2 = nn.Linear(fc_hidden_size, 1)
        
        # Advantage stream
        self.advantage_fc1 = nn.Linear(conv_output_size, fc_hidden_size)
        self.advantage_fc2 = nn.Linear(fc_hidden_size, self.num_actions)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling architecture.
        
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        # Shared convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        # Value stream
        value = F.relu(self.value_fc1(x))
        value = self.dropout(value)
        value = self.value_fc2(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.dropout(advantage)
        advantage = self.advantage_fc2(advantage)
        
        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_action(
        self, 
        state: torch.Tensor, 
        action_mask: torch.Tensor,
        epsilon: float = 0.0,
    ) -> int:
        """Select action using epsilon-greedy with masking."""
        if torch.rand(1).item() < epsilon:
            valid_actions = torch.where(action_mask[0])[0]
            if len(valid_actions) == 0:
                raise ValueError("No valid actions available!")
            return valid_actions[torch.randint(len(valid_actions), (1,))].item()
        
        with torch.no_grad():
            q_values = self.forward(state)
            masked_q_values = q_values.clone()
            masked_q_values[~action_mask] = float('-inf')
            action = masked_q_values.argmax(dim=1).item()
            return action
