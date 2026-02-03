#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Interactive Battleship client for testing.

This client allows direct interaction with the Battleship game
for human players in the terminal.
"""

import sys
from game.battleship_game import BattleshipGame


def print_help():
    """Print help message."""
    print("""
Battleship Commands:
  <coordinate>  - Fire at coordinate (e.g., A5, B10, J1)
  board         - Show current board
  status        - Show game status
  help          - Show this help
  quit          - Exit game

Coordinate format: Letter (A-J) + Number (1-10)
Examples: A1, E5, J10
""")


def main():
    """Run interactive Battleship game."""
    print("=" * 50)
    print("       BATTLESHIP")
    print("=" * 50)
    print("\nSink all 5 ships to win!")
    print("Ships: Carrier(5), Battleship(4), Cruiser(3),")
    print("       Submarine(3), Destroyer(2)")
    print("\nType 'help' for commands.\n")

    # Create game with fixed seed for reproducibility
    seed = 42
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except ValueError:
            pass
    
    game = BattleshipGame(seed=seed)
    print(f"Game initialized (seed: {seed})\n")
    
    # Show initial board
    print(game.get_board_string())
    print()

    while True:
        try:
            user_input = input("Enter coordinate (or 'help'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("quit", "exit", "q"):
            print("Thanks for playing!")
            break

        if cmd == "help":
            print_help()
            continue

        if cmd == "board":
            print()
            print(game.get_board_string())
            print()
            continue

        if cmd == "status":
            status = game.get_game_status()
            print(f"\n--- Game Status ---")
            print(f"Ships remaining: {status['ships_remaining']}/{status['total_ships']}")
            print(f"Ships sunk: {status['sunk_ships'] or 'None'}")
            print(f"Shots fired: {status['shots_fired']}")
            print(f"Hits: {status['hits']}, Misses: {status['misses']}")
            print()
            continue

        # Treat as coordinate
        result = game.make_shot(user_input)
        
        print(f"\n{result['message']}")
        
        if result['sunk']:
            print(f"*** {result['sunk']} DESTROYED! ***")
        
        print()
        print(game.get_board_string())
        print()

        if game.is_game_over():
            status = game.get_game_status()
            print("=" * 50)
            print("  VICTORY! All ships destroyed!")
            print("=" * 50)
            print(f"\nFinal stats:")
            print(f"  Shots fired: {status['shots_fired']}")
            print(f"  Hits: {status['hits']}")
            print(f"  Misses: {status['misses']}")
            print(f"  Accuracy: {status['hits']/status['shots_fired']*100:.1f}%")
            break


if __name__ == "__main__":
    main()
