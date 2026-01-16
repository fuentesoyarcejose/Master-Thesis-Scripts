import os
import sys
import pandas as pd

from stance_classify_rule_based import read_csv_robust, classify_dataframe


def main(argv):
    import argparse

    parser = argparse.ArgumentParser(description="Generate seed labels for stance training using rule-based classifier.")
    parser.add_argument("--input", default=os.path.join("Stance", "DataSample.csv"), help="Input CSV path")
    parser.add_argument("--text-col", default="text", help="Text column name (auto-fallback if missing)")
    parser.add_argument("--neutral-delta", type=float, default=0.2, help="Neutral threshold (score difference)")
    parser.add_argument("--min_conf", type=float, default=0.2, help="Minimum score gap to keep non-neutral labels as confident seeds (default: 0.2)")
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap")
    parser.add_argument("--output", default=os.path.join("Stance", "seed_labels.csv"), help="Output CSV with 'text' and 'label' columns")

    args = parser.parse_args(argv)

    df = read_csv_robust(args.input)
    if args.text_col not in df.columns:
        candidates = ["text", "post_text", "link_attachment.caption", "link_attachment.description"]
        found = next((c for c in candidates if c in df.columns), None)
        if not found:
            raise KeyError(f"Text column '{args.text_col}' not found. Available columns: {list(df.columns)}")
        args.text_col = found

    classified = classify_dataframe(df, args.text_col, neutral_delta=args.neutral_delta, limit=args.limit)

    def label_to_id(stance: str) -> int:
        if stance == "Pro-Palestine":
            return 0
        if stance == "Neutral":
            return 1
        if stance == "Pro-Israel":
            return 2
        return -1

    # Compute confidence gap and filter low-confidence non-neutral
    gap = (classified["pro_palestine_score"].fillna(0) - classified["pro_israel_score"].fillna(0)).abs()
    labels = classified["stance"].astype(str).map(label_to_id)
    keep_mask = (labels == 1) | (gap >= args.min_conf)
    seeds = classified.loc[keep_mask, [args.text_col, "stance", "pro_palestine_score", "pro_israel_score"]].copy()
    seeds["label"] = seeds["stance"].map(label_to_id)
    seeds.rename(columns={args.text_col: "text"}, inplace=True)

    seeds = seeds.loc[seeds["label"] >= 0, ["text", "label", "pro_palestine_score", "pro_israel_score"]]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    seeds.to_csv(args.output, index=False)
    print(f"Saved seed labels to: {args.output} (rows: {len(seeds)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


