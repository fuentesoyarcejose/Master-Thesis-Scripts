import os
import sys
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def read_csv_robust(csv_path: str) -> pd.DataFrame:
    """
    Read CSV robustly, preserving multiline text fields.
    Falls back across common engines/options if needed.
    """
    read_kwargs = dict(
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )
    try:
        return pd.read_csv(csv_path, engine="python", **read_kwargs)
    except Exception:
        return pd.read_csv(csv_path, engine="c", quoting=3, **read_kwargs)


def ensure_openai_client():
    """
    Lazy import and instantiate OpenAI client. Requires OPENAI_API_KEY in env.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'openai' package is required. Install via: pip install openai"
        ) from exc

    api_key = ""
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable to use OpenAI API.")

    return OpenAI(api_key=api_key)


SYSTEM_PROMPT = (
    "You are a precise stance classifier. Given a text, assess sentiment/position on the Israel–Palestine "
    "topic. Return calibrated scores in [0,1] for pro_palestine and pro_israel (independent)."
)


def build_user_prompt(text: str, threshold: float) -> str:
    return (
        "Classify stance.") + (
        "\nInstructions:" 
        "\n- Output a single JSON object only, no prose."
        "\n- Fields: pro_palestine_score (0..1), pro_israel_score (0..1), rationale (short)."
        "\n- DO NOT include a 'stance' field; it will be computed externally using the difference threshold."
        f"\n- Scores should reflect advocacy/support tone, not just mentions. Threshold for neutrality: {threshold:.3f}."
        "\n\nText:\n" + text.strip()
    )


def call_openai_classify(
    client: Any,
    model: str,
    text: str,
    threshold: float,
    max_retries: int = 3,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """
    Call OpenAI chat model to get stance scores for a single text.
    Returns dict with scores and rationale.
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(text, threshold)},
                ],
                timeout=timeout_s,
            )
            content = resp.choices[0].message.content.strip()  # type: ignore
            return json.loads(content)
        except Exception as exc:  # pragma: no cover
            last_err = exc
            # simple backoff
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"OpenAI classification failed after {max_retries} attempts: {last_err}")


def compute_stance(pro_pal: float, pro_isr: float, neutral_delta: float) -> str:
    diff = abs(pro_pal - pro_isr)
    if diff < neutral_delta:
        return "Neutral"
    return "Pro-Palestine" if pro_pal > pro_isr else "Pro-Israel"


def classify_dataframe(
    df: pd.DataFrame,
    text_col: str,
    model: str,
    neutral_delta: float,
    max_chars: int = 4000,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    client = ensure_openai_client()

    texts: List[str] = df[text_col].astype(str).tolist()
    if limit is not None:
        texts = texts[:limit]

    pro_pal_scores: List[Optional[float]] = []
    pro_isr_scores: List[Optional[float]] = []
    stances: List[Optional[str]] = []
    rationales: List[Optional[str]] = []

    for idx, t in enumerate(texts):
        text_for_model = (t or "").strip()
        if not text_for_model:
            pro_pal_scores.append(None)
            pro_isr_scores.append(None)
            stances.append(None)
            rationales.append(None)
            continue

        if len(text_for_model) > max_chars:
            text_for_model = text_for_model[:max_chars]

        try:
            result = call_openai_classify(client, model, text_for_model, neutral_delta)
            pro_pal = float(result.get("pro_palestine_score", 0.0))
            pro_isr = float(result.get("pro_israel_score", 0.0))
            stance = compute_stance(pro_pal, pro_isr, neutral_delta)
            rationale = str(result.get("rationale", "")).strip() or None
        except Exception as exc:
            pro_pal = None  # type: ignore
            pro_isr = None  # type: ignore
            stance = None  # type: ignore
            rationale = f"LLM error: {exc}"  # type: ignore

        pro_pal_scores.append(pro_pal)
        pro_isr_scores.append(pro_isr)
        stances.append(stance)
        rationales.append(rationale)

    out = df.copy()
    if limit is not None:
        out = out.iloc[:limit].copy()

    out["pro_palestine_score"] = pro_pal_scores
    out["pro_israel_score"] = pro_isr_scores
    out["stance"] = stances
    out["stance_rationale"] = rationales
    return out


def main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Classify stance (Pro-Palestine | Pro-Israel | Neutral) from CSV text using an LLM.")
    parser.add_argument("--input", default=os.path.join("Stance", "DataSample.csv"), help="Path to input CSV (default: Stance/DataSample.csv)")
    parser.add_argument("--text-col", default="text", help="Name of the text column (default: text)")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), help="OpenAI model name (default: gpt-4o-mini)")
    parser.add_argument("--neutral-delta", type=float, default=0.2, help="Max score difference to be considered Neutral (default: 0.2)")
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit number of rows (debug/cost control)")
    parser.add_argument("--max-chars", type=int, default=4000, help="Max characters of text sent to the model (default: 4000)")
    parser.add_argument("--output", default=os.path.join("Stance", "DataSample_stance.csv"), help="Output CSV path (default: Stance/DataSample_stance.csv)")

    args = parser.parse_args(argv)

    df = read_csv_robust(args.input)
    if args.text_col not in df.columns:
        # Try common alternative name heuristics
        candidates = ["text", "post_text", "link_attachment.caption", "link_attachment.description"]
        found = next((c for c in candidates if c in df.columns), None)
        if not found:
            raise KeyError(f"Text column '{args.text_col}' not found. Available columns: {list(df.columns)}")
        args.text_col = found

    result_df = classify_dataframe(
        df=df,
        text_col=args.text_col,
        model=args.model,
        neutral_delta=args.neutral_delta,
        max_chars=args.max_chars,
        limit=args.limit,
    )

    # Safe write
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"Saved stance results to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


