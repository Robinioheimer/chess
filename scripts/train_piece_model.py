from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockfish_vision.recognition.hog_svm import default_model_path, train_and_save  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Trainiert das HOG-SVM-Figurenmodell.")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = train_and_save(path=args.out, samples_per_class=args.samples)
    print(f"Modell gespeichert: {path}")
    print(f"Standardpfad: {default_model_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
