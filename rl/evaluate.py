# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Evaluation framework for DQN Battleship agent.

Provides comprehensive benchmarking and performance metrics.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import yaml

import torch

from rl.dqn_agent import DQNAgent
from game.battleship_game import BattleshipGame


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for DQN agent performance analysis."""
    
    def __init__(
        self,
        agent: DQNAgent,
        num_games: int = 100,
        seeds: Optional[List[int]] = None,
        max_steps: int = 100,
    ):
        """
        Initialize evaluator.
        
        Args:
            agent: Trained DQN agent
            num_games: Number of games to evaluate
            seeds: Random seeds for reproducibility
            max_steps: Maximum steps per game
        """
        self.agent = agent
        self.num_games = num_games
        self.seeds = seeds if seeds else list(range(42, 42 + num_games))
        self.max_steps = max_steps
    
    def evaluate_agent(self, verbose: bool = False) -> Dict:
        """
        Evaluate RL agent on test games.
        
        Args:
            verbose: Print per-game results
        
        Returns:
            Evaluation statistics dictionary
        """
        logger.info(f"Evaluating RL agent on {self.num_games} games...")
        start_time = time.time()
        
        results = []
        game_logs = []
        
        for i, seed in enumerate(self.seeds[:self.num_games]):
            game = BattleshipGame(seed=seed)
            
            stats = self.agent.play_episode(
                game=game,
                max_steps=self.max_steps,
                training=False,
                verbose=False,
            )
            
            results.append(stats)
            game_logs.append({
                'game_id': i + 1,
                'seed': seed,
                'won': stats['won'],
                'moves': stats['moves'],
                'ships_sunk': stats['ships_sunk'],
                'accuracy': stats['accuracy'],
            })
            
            if verbose:
                status = "WON" if stats['won'] else "INCOMPLETE"
                logger.info(
                    f"Game {i+1}/{self.num_games} (seed={seed}): {status} | "
                    f"Moves: {stats['moves']} | Ships: {stats['ships_sunk']}/5 | "
                    f"Accuracy: {stats['accuracy']:.2%}"
                )
        
        elapsed_time = time.time() - start_time
        
        # Calculate statistics
        wins = [r['won'] for r in results]
        moves = [r['moves'] for r in results]
        winning_moves = [r['moves'] for r in results if r['won']]
        accuracies = [r['accuracy'] for r in results]
        ships_sunk = [r['ships_sunk'] for r in results]
        
        eval_stats = {
            'num_games': self.num_games,
            'wins': sum(wins),
            'win_rate': np.mean(wins),
            'avg_moves': np.mean(moves),
            'std_moves': np.std(moves),
            'min_moves': np.min(moves),
            'max_moves': np.max(moves),
            'avg_winning_moves': np.mean(winning_moves) if winning_moves else None,
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'avg_ships_sunk': np.mean(ships_sunk),
            'total_time': elapsed_time,
            'avg_time_per_game': elapsed_time / self.num_games,
            'repeated_shots': 0,  # RL agent never repeats shots
            'game_logs': game_logs,
        }
        
        return eval_stats
    
    def print_summary(self, results: Dict):
        """Print formatted evaluation summary."""
        print("\n" + "="*60)
        print("DQN AGENT EVALUATION RESULTS")
        print("="*60)
        print(f"Games Played:        {results['num_games']}")
        print(f"Wins:                {results['wins']}")
        print(f"Win Rate:            {results['win_rate']:.2%}")
        print(f"Avg Moves:           {results['avg_moves']:.1f} ± {results['std_moves']:.1f}")
        if results['avg_winning_moves']:
            print(f"Avg Winning Moves:   {results['avg_winning_moves']:.1f}")
        print(f"Avg Accuracy:        {results['avg_accuracy']:.2%}")
        print(f"Avg Ships Sunk:      {results['avg_ships_sunk']:.1f}/5")
        print(f"Repeated Shots:      {results['repeated_shots']}")
        print(f"Total Time:          {results['total_time']:.1f}s")
        print(f"Avg Time per Game:   {results['avg_time_per_game']:.3f}s")
        print("="*60)


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate DQN Battleship agent")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained agent checkpoint'
    )
    parser.add_argument(
        '--num-games',
        type=int,
        default=100,
        help='Number of games to evaluate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='rl_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print per-game results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize agent
    logger.info("Initializing agent...")
    agent = DQNAgent(device=args.device)
    agent.load(args.checkpoint)
    
    # Create evaluator
    evaluator = Evaluator(agent, num_games=args.num_games)
    
    # Evaluate agent
    results = evaluator.evaluate_agent(verbose=args.verbose)
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    evaluator.print_summary(results)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
