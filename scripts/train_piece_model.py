from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockfish_vision.recognition.hog_svm import (  # noqa: E402
    default_model_path,
    train_and_save,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trainiert das dokumentierte HOG-SVM aus synthetischen Figuren-Glyphen."
    )
    parser.add_argument("--samples", type=int, default=64, help="Beispiele pro Klasse (mindestens 4)")
    parser.add_argument("--seed", type=int, default=7, help="Reproduzierbarer Zufalls-Seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Ziel-XML; Standard ist das gebündelte Modell im Source-Checkout",
    )
    args = parser.parse_args()
    if args.samples < 4:
        parser.error("--samples muss mindestens 4 sein")
    path = train_and_save(path=args.out, samples_per_class=args.samples, seed=args.seed)
    print(f"Modell gespeichert: {path}")
    print(f"Metadaten gespeichert: {path.with_suffix('.json')}")
    print(f"Standardpfad der Anwendung: {default_model_path()}")
    print("Hinweis: Das Modell kennt synthetische Glyphen, nicht beliebige Web-Figurensets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
