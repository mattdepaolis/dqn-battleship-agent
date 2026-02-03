# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Battleship game logic module.

Provides the core game mechanics for Battleship including:
- Board representation and rendering
- Ship placement (random or deterministic)
- Shot handling with hit/miss/sunk detection
- Game state tracking
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class CellState(Enum):
    """State of a cell on the board."""
    UNEXPLORED = "_"
    HIT = "X"
    MISS = "O"


@dataclass
class Ship:
    """Represents a ship on the board."""
    name: str
    size: int
    positions: List[Tuple[int, int]] = field(default_factory=list)
    hits: set = field(default_factory=set)

    @property
    def is_sunk(self) -> bool:
        """Check if all positions have been hit."""
        return len(self.hits) >= self.size

    def hit(self, position: Tuple[int, int]) -> bool:
        """Record a hit at the given position. Returns True if valid hit."""
        if position in self.positions and position not in self.hits:
            self.hits.add(position)
            return True
        return False


# Standard Battleship ships
STANDARD_SHIPS = [
    ("Carrier", 5),
    ("Battleship", 4),
    ("Cruiser", 3),
    ("Submarine", 3),
    ("Destroyer", 2),
]


class BattleshipGame:
    """
    Battleship game implementation.
    
    The game board is 10x10 with rows labeled A-J and columns 1-10.
    Ships are placed and the player makes shots to sink all ships.
    """

    BOARD_SIZE = 10
    ROW_LABELS = "ABCDEFGHIJ"

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize a new Battleship game.
        
        Args:
            seed: Random seed for reproducible ship placement.
        """
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Board tracks what the player can see (their shots)
        self.player_view: List[List[CellState]] = [
            [CellState.UNEXPLORED for _ in range(self.BOARD_SIZE)]
            for _ in range(self.BOARD_SIZE)
        ]
        
        # Ships on the board
        self.ships: List[Ship] = []
        
        # Track all ship positions for quick lookup
        self._ship_positions: Dict[Tuple[int, int], Ship] = {}
        
        # Game statistics
        self.shots_fired = 0
        self.hits = 0
        self.misses = 0
        
        # Place ships
        self._place_ships()

    def _place_ships(self) -> None:
        """Place all standard ships randomly on the board."""
        for ship_name, ship_size in STANDARD_SHIPS:
            ship = Ship(name=ship_name, size=ship_size)
            placed = False
            
            attempts = 0
            max_attempts = 1000
            
            while not placed and attempts < max_attempts:
                attempts += 1
                
                # Random starting position
                row = self.rng.randint(0, self.BOARD_SIZE - 1)
                col = self.rng.randint(0, self.BOARD_SIZE - 1)
                
                # Random direction: 0 = horizontal, 1 = vertical
                horizontal = self.rng.choice([True, False])
                
                # Calculate positions
                positions = []
                valid = True
                
                for i in range(ship_size):
                    if horizontal:
                        new_row, new_col = row, col + i
                    else:
                        new_row, new_col = row + i, col
                    
                    # Check bounds
                    if not (0 <= new_row < self.BOARD_SIZE and 
                            0 <= new_col < self.BOARD_SIZE):
                        valid = False
                        break
                    
                    # Check collision
                    if (new_row, new_col) in self._ship_positions:
                        valid = False
                        break
                    
                    positions.append((new_row, new_col))
                
                if valid:
                    ship.positions = positions
                    for pos in positions:
                        self._ship_positions[pos] = ship
                    self.ships.append(ship)
                    placed = True
            
            if not placed:
                raise RuntimeError(f"Failed to place ship {ship_name}")

    def place_ships_deterministic(self, placements: List[dict]) -> None:
        """
        Place ships at specific positions (for reproducible games).
        
        Args:
            placements: List of dicts with 'name', 'start' (e.g., 'A1'), 
                       'horizontal' (bool)
        """
        # Clear existing placements
        self.ships.clear()
        self._ship_positions.clear()
        
        ship_configs = {name: size for name, size in STANDARD_SHIPS}
        
        for placement in placements:
            name = placement["name"]
            start = placement["start"]
            horizontal = placement["horizontal"]
            
            if name not in ship_configs:
                raise ValueError(f"Unknown ship: {name}")
            
            size = ship_configs[name]
            start_row, start_col = self.parse_coordinate(start)
            
            ship = Ship(name=name, size=size)
            positions = []
            
            for i in range(size):
                if horizontal:
                    pos = (start_row, start_col + i)
                else:
                    pos = (start_row + i, start_col)
                
                if not (0 <= pos[0] < self.BOARD_SIZE and 
                        0 <= pos[1] < self.BOARD_SIZE):
                    raise ValueError(f"Ship {name} goes out of bounds")
                
                if pos in self._ship_positions:
                    raise ValueError(f"Ship {name} overlaps with another ship")
                
                positions.append(pos)
            
            ship.positions = positions
            for pos in positions:
                self._ship_positions[pos] = ship
            self.ships.append(ship)

    def parse_coordinate(self, coord: str) -> Tuple[int, int]:
        """
        Parse a coordinate string like 'A5' or 'B10' into (row, col).
        
        Args:
            coord: Coordinate string (e.g., 'A5', 'J10')
            
        Returns:
            Tuple of (row_index, col_index), both 0-based.
            
        Raises:
            ValueError: If coordinate is invalid.
        """
        coord = coord.strip().upper()
        
        if len(coord) < 2 or len(coord) > 3:
            raise ValueError(f"Invalid coordinate format: {coord}")
        
        row_char = coord[0]
        col_str = coord[1:]
        
        if row_char not in self.ROW_LABELS:
            raise ValueError(f"Invalid row '{row_char}'. Must be A-J.")
        
        try:
            col_num = int(col_str)
        except ValueError:
            raise ValueError(f"Invalid column '{col_str}'. Must be 1-10.")
        
        if not (1 <= col_num <= 10):
            raise ValueError(f"Column {col_num} out of range. Must be 1-10.")
        
        row_index = self.ROW_LABELS.index(row_char)
        col_index = col_num - 1  # Convert to 0-based
        
        return (row_index, col_index)

    def format_coordinate(self, row: int, col: int) -> str:
        """Convert (row, col) indices to coordinate string like 'A5'."""
        return f"{self.ROW_LABELS[row]}{col + 1}"

    def make_shot(self, coord: str) -> dict:
        """
        Make a shot at the given coordinate.
        
        Args:
            coord: Target coordinate (e.g., 'A5')
            
        Returns:
            Dict with keys:
                - 'valid': bool, whether shot was valid
                - 'result': 'hit', 'miss', or 'already_shot'
                - 'coordinate': the coordinate shot at
                - 'sunk': ship name if a ship was sunk, None otherwise
                - 'message': human-readable result message
        """
        try:
            row, col = self.parse_coordinate(coord)
        except ValueError as e:
            return {
                "valid": False,
                "result": "invalid",
                "coordinate": coord,
                "sunk": None,
                "message": str(e)
            }
        
        # Check if already shot
        if self.player_view[row][col] != CellState.UNEXPLORED:
            return {
                "valid": False,
                "result": "already_shot",
                "coordinate": coord,
                "sunk": None,
                "message": f"Already shot at {coord}."
            }
        
        self.shots_fired += 1
        position = (row, col)
        
        # Check for hit
        if position in self._ship_positions:
            ship = self._ship_positions[position]
            ship.hit(position)
            self.player_view[row][col] = CellState.HIT
            self.hits += 1
            
            if ship.is_sunk:
                return {
                    "valid": True,
                    "result": "hit",
                    "coordinate": coord,
                    "sunk": ship.name,
                    "message": f"Hit! You sunk the {ship.name}!"
                }
            else:
                return {
                    "valid": True,
                    "result": "hit",
                    "coordinate": coord,
                    "sunk": None,
                    "message": "Hit!"
                }
        else:
            self.player_view[row][col] = CellState.MISS
            self.misses += 1
            return {
                "valid": True,
                "result": "miss",
                "coordinate": coord,
                "sunk": None,
                "message": "Miss!"
            }

    def get_board_string(self) -> str:
        """
        Get the current board state as an ASCII string.
        
        Returns:
            ASCII representation of the board showing player's view.
        """
        lines = []
        
        # Header row
        header = "    |" + "|".join(f"{i:^3}" for i in range(1, 11))
        lines.append(header)
        
        # Board rows
        for row_idx, row in enumerate(self.player_view):
            row_label = self.ROW_LABELS[row_idx]
            cells = "|".join(f" {cell.value} " for cell in row)
            lines.append(f"  {row_label} |{cells}")
        
        return "\n".join(lines)

    def get_game_status(self) -> dict:
        """
        Get current game status.
        
        Returns:
            Dict with game statistics and status.
        """
        ships_remaining = sum(1 for ship in self.ships if not ship.is_sunk)
        ships_sunk = len(self.ships) - ships_remaining
        
        return {
            "game_over": ships_remaining == 0,
            "ships_remaining": ships_remaining,
            "ships_sunk": ships_sunk,
            "total_ships": len(self.ships),
            "shots_fired": self.shots_fired,
            "hits": self.hits,
            "misses": self.misses,
            "sunk_ships": [ship.name for ship in self.ships if ship.is_sunk]
        }

    def is_game_over(self) -> bool:
        """Check if all ships have been sunk."""
        return all(ship.is_sunk for ship in self.ships)

    def get_ship_placements(self) -> List[dict]:
        """
        Get current ship placements for serialization.
        
        Returns:
            List of placement dicts that can be used with place_ships_deterministic.
        """
        placements = []
        for ship in self.ships:
            if ship.positions:
                start_row, start_col = ship.positions[0]
                # Determine if horizontal
                horizontal = True
                if len(ship.positions) > 1:
                    horizontal = ship.positions[1][0] == ship.positions[0][0]
                
                placements.append({
                    "name": ship.name,
                    "start": self.format_coordinate(start_row, start_col),
                    "horizontal": horizontal
                })
        return placements
