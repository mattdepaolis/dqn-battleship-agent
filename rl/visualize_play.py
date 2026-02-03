#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Visualize DQN agent playing Battleship in the terminal.

Shows board evolution, Q-values, and decision-making process.
"""

import argparse
import time
import torch
import numpy as np
from typing import Optional

from rl.dqn_agent import DQNAgent
from game.battleship_game import BattleshipGame, CellState


class GameVisualizer:
    """Visualize DQN agent gameplay in terminal."""
    
    def __init__(self, agent: DQNAgent, delay: float = 1.0):
        """
        Initialize visualizer.
        
        Args:
            agent: Trained DQN agent
            delay: Delay between moves in seconds (0 = wait for Enter)
        """
        self.agent = agent
        self.delay = delay
    
    def print_board(self, game: BattleshipGame, title: str = ""):
        """Print formatted game board."""
        if title:
            print(f"\n{'='*60}")
            print(f"  {title}")
            print('='*60)
        
        print(game.get_board_string())
        
        stats = game.get_game_status()
        print(f"\n  Shots: {stats['shots_fired']} | "
              f"Hits: {stats['hits']} | "
              f"Misses: {stats['misses']} | "
              f"Ships Sunk: {stats['ships_sunk']}/5")
        
        if stats['sunk_ships']:
            print(f"  Sunk Ships: {', '.join(stats['sunk_ships'])}")
    
    def print_q_values_summary(
        self, 
        q_values: torch.Tensor, 
        action_mask: torch.Tensor,
        selected_action: int,
        top_k: int = 5,
    ):
        """Print summary of top Q-values."""
        # Get valid Q-values
        masked_q = q_values.clone()
        masked_q[~action_mask] = float('-inf')
        
        # Get top-k actions
        valid_q = masked_q[0][action_mask[0]]
        valid_actions = torch.where(action_mask[0])[0]
        
        if len(valid_actions) > 0:
            top_k = min(top_k, len(valid_actions))
            top_values, top_indices = torch.topk(valid_q, top_k)
            top_actions = valid_actions[top_indices]
            
            print(f"\n  Top {top_k} Q-Values:")
            for i, (action, q_val) in enumerate(zip(top_actions, top_values)):
                coord = self.agent.action_to_coordinate(action.item())
                marker = "→" if action.item() == selected_action else " "
                print(f"    {marker} {coord:>4}: {q_val.item():>8.2f}")
    
    def print_move_summary(
        self,
        move_num: int,
        coordinate: str,
        result: dict,
        q_value: float,
    ):
        """Print summary of move taken."""
        print(f"\n  Move #{move_num}: {coordinate}")
        print(f"  Q-Value: {q_value:.2f}")
        print(f"  Result: {result['message']}")
        
        if result.get('sunk'):
            print(f"  🎯 SUNK: {result['sunk']}!")
    
    def visualize_game(
        self,
        game: Optional[BattleshipGame] = None,
        seed: Optional[int] = None,
        max_moves: int = 100,
        show_q_values: bool = True,
        interactive: bool = False,
    ):
        """
        Visualize agent playing a complete game.
        
        Args:
            game: BattleshipGame instance (creates new if None)
            seed: Random seed for reproducibility
            max_moves: Maximum moves before giving up
            show_q_values: Whether to show Q-value rankings
            interactive: Whether to wait for Enter between moves
        """
        if game is None:
            game = BattleshipGame(seed=seed)
        
        # Print initial board
        print("\n" + "="*60)
        print("  DQN AGENT PLAYING BATTLESHIP")
        print("="*60)
        print(f"\n  Model: DQN with Action Masking")
        print(f"  Seed: {seed if seed else 'Random'}")
        print(f"  Device: {self.agent.device}")
        
        if interactive:
            print("\n  Press Enter to advance...")
        else:
            print(f"  Delay: {self.delay}s between moves")
        
        self.print_board(game, "INITIAL BOARD")
        
        if interactive:
            input("\n  Press Enter to start...")
        else:
            time.sleep(self.delay * 2)
        
        # Play game
        move_num = 0
        
        for step in range(max_moves):
            move_num += 1
            
            # Get current state
            state, action_mask = self.agent.state_to_tensor(game)
            
            # Get Q-values
            with torch.no_grad():
                q_values = self.agent.policy_net(state)
            
            # Select action (greedy - no exploration)
            action = self.agent.select_action(state, action_mask, training=False)
            coordinate = self.agent.action_to_coordinate(action)
            selected_q_value = q_values[0, action].item()
            
            # Show Q-values if requested
            if show_q_values:
                self.print_q_values_summary(q_values, action_mask, action, top_k=5)
            
            # Take action
            result = game.make_shot(coordinate)
            
            if not result['valid']:
                print(f"\n  ❌ ERROR: Invalid action {coordinate}")
                break
            
            # Print move summary
            self.print_move_summary(move_num, coordinate, result, selected_q_value)
            
            # Show updated board
            self.print_board(game, f"AFTER MOVE {move_num}")
            
            # Check if won
            if game.is_game_over():
                print("\n" + "="*60)
                print("  🎉 GAME WON! ALL SHIPS DESTROYED!")
                print("="*60)
                
                stats = game.get_game_status()
                accuracy = stats['hits'] / stats['shots_fired'] * 100
                
                print(f"\n  Final Stats:")
                print(f"    Total Moves: {move_num}")
                print(f"    Accuracy: {accuracy:.1f}%")
                print(f"    Ships Sunk: {stats['ships_sunk']}/5")
                print("\n")
                break
            
            # Wait for next move
            if interactive:
                input("\n  Press Enter for next move...")
            else:
                time.sleep(self.delay)
        else:
            print("\n" + "="*60)
            print("  ⏱️  GAME INCOMPLETE - Max moves reached")
            print("="*60)
    
    def compare_strategies(
        self,
        game: BattleshipGame,
        move_num: int = 1,
    ):
        """
        Show what the agent considers for a single move.
        
        Displays all valid actions ranked by Q-value.
        """
        state, action_mask = self.agent.state_to_tensor(game)
        
        with torch.no_grad():
            q_values = self.agent.policy_net(state)
        
        # Get all valid actions and their Q-values
        valid_mask = action_mask[0]
        valid_indices = torch.where(valid_mask)[0]
        valid_q_values = q_values[0][valid_mask]
        
        # Sort by Q-value
        sorted_indices = torch.argsort(valid_q_values, descending=True)
        
        print(f"\n{'='*60}")
        print(f"  MOVE {move_num} - ALL VALID ACTIONS (Ranked by Q-Value)")
        print('='*60)
        
        print(f"\n  Board State:")
        self.print_board(game, "")
        
        print(f"\n  Valid Actions: {len(valid_indices)}")
        print(f"\n  {'Rank':<6} {'Coord':<6} {'Q-Value':<12} {'Row':<6} {'Strategy'}")
        print("  " + "-"*56)
        
        for rank, idx in enumerate(sorted_indices[:20], 1):  # Show top 20
            action = valid_indices[idx].item()
            coord = self.agent.action_to_coordinate(action)
            q_val = valid_q_values[idx].item()
            
            row_label = coord[0]
            col = int(coord[1:])
            
            # Infer strategy
            strategy = "Random"
            if q_val > valid_q_values.mean().item():
                strategy = "Promising"
            if rank <= 3:
                strategy = "Top Choice"
            
            print(f"  {rank:<6} {coord:<6} {q_val:<12.2f} {row_label:<6} {strategy}")
        
        if len(valid_indices) > 20:
            print(f"\n  ... and {len(valid_indices) - 20} more actions")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize DQN agent playing Battleship"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/dqn_agent_final.pt',
        help='Path to trained agent checkpoint'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for game'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between moves in seconds (0 = interactive mode)'
    )
    parser.add_argument(
        '--no-q-values',
        action='store_true',
        help='Hide Q-value rankings'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Wait for Enter between moves (ignores --delay)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )
    parser.add_argument(
        '--analyze-move',
        type=int,
        help='Analyze a specific move (shows all Q-values)'
    )
    
    args = parser.parse_args()
    
    # Load agent
    print("Loading DQN agent...")
    agent = DQNAgent(device=args.device)
    agent.load(args.checkpoint)
    print(f"Agent loaded from {args.checkpoint}")
    
    # Create visualizer
    visualizer = GameVisualizer(agent, delay=args.delay)
    
    # Run visualization
    if args.analyze_move:
        # Analyze specific move
        game = BattleshipGame(seed=args.seed)
        
        # Play up to that move
        for _ in range(args.analyze_move - 1):
            state, action_mask = agent.state_to_tensor(game)
            action = agent.select_action(state, action_mask, training=False)
            coord = agent.action_to_coordinate(action)
            game.make_shot(coord)
        
        visualizer.compare_strategies(game, args.analyze_move)
    else:
        # Play full game with visualization
        visualizer.visualize_game(
            seed=args.seed,
            show_q_values=not args.no_q_values,
            interactive=args.interactive or args.delay == 0,
        )


if __name__ == "__main__":
    main()
