from __future__ import annotations

from stockfish_vision.policy import (
    CONFIRMATION_REQUIRED,
    LIVE_GAME_BLOCK_MESSAGE,
    LiveGameGuard,
    RAPID_CHANGE_MESSAGE,
    can_analyze,
)


def test_training_and_pgn_allowed() -> None:
    ok, _ = can_analyze("training")
    assert ok
    ok, _ = can_analyze("pgn_completed")
    assert ok


def test_live_mode_blocked() -> None:
    ok, message = can_analyze("live_online")
    assert not ok
    assert message == LIVE_GAME_BLOCK_MESSAGE


def test_snapshot_needs_confirmation() -> None:
    ok, message = can_analyze("frozen_snapshot", snapshot_confirmed=False)
    assert not ok
    assert message == CONFIRMATION_REQUIRED
    ok, _ = can_analyze("frozen_snapshot", snapshot_confirmed=True)
    assert ok


def test_rapid_changes_look_like_live_game() -> None:
    guard = LiveGameGuard()
    guard.record("pos1")
    guard.record("pos2")
    guard.record("pos3")
    assert guard.looks_like_live_game()
    ok, message = can_analyze(
        "frozen_snapshot", snapshot_confirmed=True, live_guard=guard
    )
    assert not ok
    assert message == RAPID_CHANGE_MESSAGE
