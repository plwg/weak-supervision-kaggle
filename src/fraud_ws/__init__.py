import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weakly supervised fraud detection (Snorkel-style label model)"
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="re-download/rebuild features"
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="skip the fully supervised upper bound",
    )
    args = parser.parse_args()

    from fraud_ws import data, pipeline

    data.prepare(force=args.rebuild)
    pipeline.run(with_reference=not args.no_reference)
