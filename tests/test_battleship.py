# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for Battleship game logic.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from game.battleship_game import BattleshipGame, Ship, CellState, STANDARD_SHIPS


class TestShip:
    """Tests for Ship class."""

    def test_ship_creation(self):
        """Test creating a ship."""
        ship = Ship(name="TestShip", size=3)
        assert ship.name == "TestShip"
        assert ship.size == 3
        assert ship.positions == []
        assert len(ship.hits) == 0
        assert not ship.is_sunk

    def test_ship_hit(self):
        """Test hitting a ship."""
        ship = Ship(name="TestShip", size=2, positions=[(0, 0), (0, 1)])
        
        # First hit
        assert ship.hit((0, 0)) is True
        assert len(ship.hits) == 1
        assert not ship.is_sunk
        
        # Duplicate hit should fail
        assert ship.hit((0, 0)) is False
        assert len(ship.hits) == 1
        
        # Second hit sinks it
        assert ship.hit((0, 1)) is True
        assert ship.is_sunk

    def test_ship_hit_invalid_position(self):
        """Test hitting a position not on the ship."""
        ship = Ship(name="TestShip", size=2, positions=[(0, 0), (0, 1)])
        assert ship.hit((5, 5)) is False


class TestBattleshipGame:
    """Tests for BattleshipGame class."""

    def test_game_creation(self):
        """Test creating a new game."""
        game = BattleshipGame(seed=42)
        
        assert game.BOARD_SIZE == 10
        assert len(game.ships) == 5
        assert game.shots_fired == 0
        assert game.hits == 0
        assert game.misses == 0

    def test_game_deterministic_with_seed(self):
        """Test that same seed produces same game."""
        game1 = BattleshipGame(seed=123)
        game2 = BattleshipGame(seed=123)
        
        placements1 = game1.get_ship_placements()
        placements2 = game2.get_ship_placements()
        
        assert placements1 == placements2

    def test_parse_coordinate_valid(self):
        """Test parsing valid coordinates."""
        game = BattleshipGame(seed=0)
        
        assert game.parse_coordinate("A1") == (0, 0)
        assert game.parse_coordinate("A10") == (0, 9)
        assert game.parse_coordinate("J1") == (9, 0)
        assert game.parse_coordinate("J10") == (9, 9)
        assert game.parse_coordinate("E5") == (4, 4)
        
        # Case insensitive
        assert game.parse_coordinate("a1") == (0, 0)
        assert game.parse_coordinate("e5") == (4, 4)
        
        # With whitespace
        assert game.parse_coordinate(" B3 ") == (1, 2)

    def test_parse_coordinate_invalid(self):
        """Test parsing invalid coordinates."""
        game = BattleshipGame(seed=0)
        
        with pytest.raises(ValueError):
            game.parse_coordinate("K1")  # Invalid row
        
        with pytest.raises(ValueError):
            game.parse_coordinate("A0")  # Column out of range
        
        with pytest.raises(ValueError):
            game.parse_coordinate("A11")  # Column out of range
        
        with pytest.raises(ValueError):
            game.parse_coordinate("AA")  # Invalid column
        
        with pytest.raises(ValueError):
            game.parse_coordinate("")  # Empty

    def test_format_coordinate(self):
        """Test formatting coordinates."""
        game = BattleshipGame(seed=0)
        
        assert game.format_coordinate(0, 0) == "A1"
        assert game.format_coordinate(0, 9) == "A10"
        assert game.format_coordinate(9, 0) == "J1"
        assert game.format_coordinate(4, 4) == "E5"

    def test_make_shot_miss(self):
        """Test making a shot that misses."""
        game = BattleshipGame(seed=42)
        
        # Find an empty cell
        for row in range(10):
            for col in range(10):
                if (row, col) not in game._ship_positions:
                    coord = game.format_coordinate(row, col)
                    result = game.make_shot(coord)
                    
                    assert result["valid"] is True
                    assert result["result"] == "miss"
                    assert result["sunk"] is None
                    assert game.misses == 1
                    assert game.player_view[row][col] == CellState.MISS
                    return
        
        pytest.fail("No empty cells found")

    def test_make_shot_hit(self):
        """Test making a shot that hits."""
        game = BattleshipGame(seed=42)
        
        # Find a ship cell
        pos = list(game._ship_positions.keys())[0]
        coord = game.format_coordinate(pos[0], pos[1])
        result = game.make_shot(coord)
        
        assert result["valid"] is True
        assert result["result"] == "hit"
        assert game.hits == 1
        assert game.player_view[pos[0]][pos[1]] == CellState.HIT

    def test_make_shot_already_shot(self):
        """Test shooting at same coordinate twice."""
        game = BattleshipGame(seed=42)
        
        result1 = game.make_shot("A1")
        assert result1["valid"] is True
        
        result2 = game.make_shot("A1")
        assert result2["valid"] is False
        assert result2["result"] == "already_shot"

    def test_make_shot_invalid_coordinate(self):
        """Test shooting at invalid coordinate."""
        game = BattleshipGame(seed=42)
        
        result = game.make_shot("Z99")
        assert result["valid"] is False
        assert result["result"] == "invalid"

    def test_sink_ship(self):
        """Test sinking a complete ship."""
        game = BattleshipGame(seed=42)
        
        # Find the destroyer (2 cells) and sink it
        destroyer = None
        for ship in game.ships:
            if ship.name == "Destroyer":
                destroyer = ship
                break
        
        assert destroyer is not None
        
        # Hit all positions
        sunk_result = None
        for pos in destroyer.positions:
            coord = game.format_coordinate(pos[0], pos[1])
            result = game.make_shot(coord)
            if result["sunk"]:
                sunk_result = result
        
        assert sunk_result is not None
        assert sunk_result["sunk"] == "Destroyer"
        assert destroyer.is_sunk

    def test_game_over(self):
        """Test game over detection."""
        game = BattleshipGame(seed=42)
        
        assert not game.is_game_over()
        
        # Sink all ships
        for ship in game.ships:
            for pos in ship.positions:
                coord = game.format_coordinate(pos[0], pos[1])
                game.make_shot(coord)
        
        assert game.is_game_over()

    def test_get_board_string(self):
        """Test board string generation."""
        game = BattleshipGame(seed=42)
        board = game.get_board_string()
        
        # Check header
        assert "| 1 |" in board
        assert "|10" in board
        
        # Check rows
        assert "  A |" in board
        assert "  J |" in board
        
        # Initially all unexplored
        assert " _ " in board

    def test_get_game_status(self):
        """Test game status reporting."""
        game = BattleshipGame(seed=42)
        
        status = game.get_game_status()
        
        assert status["game_over"] is False
        assert status["ships_remaining"] == 5
        assert status["ships_sunk"] == 0
        assert status["total_ships"] == 5
        assert status["shots_fired"] == 0
        assert status["hits"] == 0
        assert status["misses"] == 0
        assert status["sunk_ships"] == []

    def test_deterministic_placement(self):
        """Test placing ships at specific positions."""
        game = BattleshipGame(seed=0)
        
        placements = [
            {"name": "Carrier", "start": "A1", "horizontal": True},
            {"name": "Battleship", "start": "B1", "horizontal": True},
            {"name": "Cruiser", "start": "C1", "horizontal": True},
            {"name": "Submarine", "start": "D1", "horizontal": True},
            {"name": "Destroyer", "start": "E1", "horizontal": True},
        ]
        
        game.place_ships_deterministic(placements)
        
        # Verify placements
        assert len(game.ships) == 5
        
        # Carrier at A1-A5
        carrier = next(s for s in game.ships if s.name == "Carrier")
        assert carrier.positions[0] == (0, 0)  # A1
        assert carrier.positions[-1] == (0, 4)  # A5

    def test_deterministic_placement_overlap_error(self):
        """Test that overlapping placements raise error."""
        game = BattleshipGame(seed=0)
        
        placements = [
            {"name": "Carrier", "start": "A1", "horizontal": True},
            {"name": "Battleship", "start": "A1", "horizontal": False},  # Overlaps!
        ]
        
        with pytest.raises(ValueError, match="overlaps"):
            game.place_ships_deterministic(placements)

    def test_deterministic_placement_out_of_bounds(self):
        """Test that out-of-bounds placements raise error."""
        game = BattleshipGame(seed=0)
        
        placements = [
            {"name": "Carrier", "start": "A8", "horizontal": True},  # Goes to A12!
        ]
        
        with pytest.raises(ValueError, match="out of bounds"):
            game.place_ships_deterministic(placements)


class TestStandardShips:
    """Tests for standard ship configuration."""

    def test_standard_ships_count(self):
        """Test that we have 5 standard ships."""
        assert len(STANDARD_SHIPS) == 5

    def test_standard_ships_total_cells(self):
        """Test total cells covered by ships."""
        total = sum(size for _, size in STANDARD_SHIPS)
        assert total == 17  # 5+4+3+3+2

    def test_standard_ships_names(self):
        """Test ship names are correct."""
        names = [name for name, _ in STANDARD_SHIPS]
        assert "Carrier" in names
        assert "Battleship" in names
        assert "Cruiser" in names
        assert "Submarine" in names
        assert "Destroyer" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
