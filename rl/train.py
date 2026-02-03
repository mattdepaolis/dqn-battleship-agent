# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Training loop for DQN Battleship agent.

Includes logging, checkpointing, and periodic evaluation.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List
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


class Trainer:
    """Trainer for DQN Battleship agent."""
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        
        # Create output directories
        self.checkpoint_dir = Path(config['output']['checkpoint_dir'])
        self.log_dir = Path(config['output']['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize agent
        self.agent = DQNAgent(
            board_size=config['game']['board_size'],
            device=config['training']['device'],
            learning_rate=config['training']['learning_rate'],
            gamma=config['training']['gamma'],
            epsilon_start=config['training']['epsilon_start'],
            epsilon_end=config['training']['epsilon_end'],
            epsilon_decay=config['training']['epsilon_decay'],
            batch_size=config['training']['batch_size'],
            buffer_size=config['training']['buffer_size'],
            target_update_freq=config['training']['target_update_freq'],
            use_double_dqn=config['training']['use_double_dqn'],
            use_dueling=config['training']['use_dueling'],
            use_prioritized_replay=config['training']['use_prioritized_replay'],
        )
        
        # Training parameters
        self.num_episodes = config['training']['num_episodes']
        self.eval_frequency = config['training']['eval_frequency']
        self.checkpoint_frequency = config['training']['checkpoint_frequency']
        self.log_frequency = config['training']['log_frequency']
        
        # Evaluation parameters
        self.eval_episodes = config['evaluation']['num_episodes']
        self.eval_seeds = config['evaluation'].get('seeds', list(range(42, 42 + self.eval_episodes)))
        
        # Training history
        self.train_history = []
        self.eval_history = []
        
        logger.info("Trainer initialized")
        logger.info(f"Training for {self.num_episodes} episodes")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Log directory: {self.log_dir}")
    
    def train(self):
        """Run training loop."""
        logger.info("Starting training...")
        start_time = time.time()
        
        episode_rewards = []
        episode_losses = []
        episode_stats = []
        
        for episode in range(1, self.num_episodes + 1):
            # Play episode
            game = BattleshipGame()
            stats = self.agent.play_episode(
                game=game,
                max_steps=self.config['game']['max_steps'],
                training=True,
                verbose=False,
            )
            
            episode_rewards.append(stats['reward'])
            episode_losses.append(stats['loss'])
            episode_stats.append(stats)
            
            # Log progress
            if episode % self.log_frequency == 0:
                avg_reward = np.mean(episode_rewards[-self.log_frequency:])
                avg_loss = np.mean(episode_losses[-self.log_frequency:])
                avg_moves = np.mean([s['moves'] for s in episode_stats[-self.log_frequency:]])
                win_rate = np.mean([s['won'] for s in episode_stats[-self.log_frequency:]])
                avg_accuracy = np.mean([s['accuracy'] for s in episode_stats[-self.log_frequency:]])
                
                logger.info(
                    f"Episode {episode}/{self.num_episodes} | "
                    f"Reward: {avg_reward:.1f} | Loss: {avg_loss:.4f} | "
                    f"Moves: {avg_moves:.1f} | Win Rate: {win_rate:.2%} | "
                    f"Accuracy: {avg_accuracy:.2%} | Epsilon: {self.agent.epsilon:.3f}"
                )
                
                # Save to history
                self.train_history.append({
                    'episode': episode,
                    'avg_reward': avg_reward,
                    'avg_loss': avg_loss,
                    'avg_moves': avg_moves,
                    'win_rate': win_rate,
                    'avg_accuracy': avg_accuracy,
                    'epsilon': self.agent.epsilon,
                })
            
            # Evaluate
            if episode % self.eval_frequency == 0:
                eval_stats = self.evaluate()
                self.eval_history.append({
                    'episode': episode,
                    **eval_stats
                })
                
                logger.info(
                    f"Evaluation @ Episode {episode} | "
                    f"Win Rate: {eval_stats['win_rate']:.2%} | "
                    f"Avg Moves: {eval_stats['avg_moves']:.1f} | "
                    f"Avg Accuracy: {eval_stats['avg_accuracy']:.2%}"
                )
            
            # Save checkpoint
            if episode % self.checkpoint_frequency == 0:
                self.save_checkpoint(episode)
        
        # Final evaluation
        logger.info("Training complete! Running final evaluation...")
        final_eval_stats = self.evaluate()
        self.eval_history.append({
            'episode': self.num_episodes,
            **final_eval_stats
        })
        
        # Save final checkpoint
        self.save_checkpoint(self.num_episodes, final=True)
        
        # Save training history
        self.save_history()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time/3600:.2f} hours")
        logger.info(f"Final Win Rate: {final_eval_stats['win_rate']:.2%}")
        logger.info(f"Final Avg Moves: {final_eval_stats['avg_moves']:.1f}")
        
        return {
            'train_history': self.train_history,
            'eval_history': self.eval_history,
            'final_stats': final_eval_stats,
            'training_time': elapsed_time,
        }
    
    def evaluate(self) -> Dict:
        """
        Evaluate agent on fixed set of games.
        
        Returns:
            Dictionary of evaluation statistics
        """
        results = []
        
        for seed in self.eval_seeds[:self.eval_episodes]:
            game = BattleshipGame(seed=seed)
            stats = self.agent.play_episode(
                game=game,
                max_steps=self.config['game']['max_steps'],
                training=False,
                verbose=False,
            )
            results.append(stats)
        
        # Aggregate statistics
        eval_stats = {
            'num_games': len(results),
            'win_rate': np.mean([r['won'] for r in results]),
            'avg_moves': np.mean([r['moves'] for r in results]),
            'avg_reward': np.mean([r['reward'] for r in results]),
            'avg_accuracy': np.mean([r['accuracy'] for r in results]),
            'avg_ships_sunk': np.mean([r['ships_sunk'] for r in results]),
            'wins': sum([r['won'] for r in results]),
        }
        
        return eval_stats
    
    def save_checkpoint(self, episode: int, final: bool = False):
        """Save training checkpoint."""
        suffix = 'final' if final else f'episode_{episode}'
        checkpoint_path = self.checkpoint_dir / f"dqn_agent_{suffix}.pt"
        self.agent.save(str(checkpoint_path))
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_history(self):
        """Save training and evaluation history."""
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump({
                'config': self.config,
                'train_history': self.train_history,
                'eval_history': self.eval_history,
            }, f, indent=2)
        logger.info(f"History saved: {history_path}")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train DQN Battleship agent")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/rl_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (overrides config)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of episodes to train (overrides config)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.device is not None:
        config['training']['device'] = args.device
    if args.episodes is not None:
        config['training']['num_episodes'] = args.episodes
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.agent.load(args.checkpoint)
        logger.info(f"Resumed from checkpoint: {args.checkpoint}")
    
    # Train
    results = trainer.train()
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final Win Rate: {results['final_stats']['win_rate']:.2%}")
    print(f"Final Avg Moves: {results['final_stats']['avg_moves']:.1f}")
    print(f"Final Avg Accuracy: {results['final_stats']['avg_accuracy']:.2%}")
    print(f"Training Time: {results['training_time']/3600:.2f} hours")
    print("="*60)


if __name__ == "__main__":
    main()
