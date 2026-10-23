#!/usr/bin/env python3
"""Minimal UCI engine for automated tests. Not Stockfish."""

from __future__ import annotations

import sys


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        command = line.strip()
        if command == "uci":
            print("id name FakeVisionEngine")
            print("id author StockfishVisionTests")
            print("option name Threads type spin default 1 min 1 max 64")
            print("option name Hash type spin default 16 min 1 max 4096")
            print("option name MultiPV type spin default 1 min 1 max 5")
            print("uciok")
        elif command == "isready":
            print("readyok")
        elif command == "ucinewgame":
            continue
        elif command.startswith("setoption"):
            continue
        elif command.startswith("position"):
            continue
        elif command.startswith("go"):
            print(
                "info depth 8 seldepth 10 multipv 1 score cp 32 nodes 1200 "
                "pv e2e4 e7e5 g1f3"
            )
            print(
                "info depth 8 seldepth 10 multipv 2 score cp 20 nodes 1200 "
                "pv d2d4 d7d5"
            )
            print("bestmove e2e4 ponder e7e5")
        elif command == "stop":
            print("bestmove e2e4")
        elif command == "quit":
            break


if __name__ == "__main__":
    main()
