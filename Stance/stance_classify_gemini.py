import os
import sys
import json
import time
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from google import genai
from google.genai import types

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# You can set this env var or hardcode it (though hardcoding is discouraged).
# Using the key found in the notebook for convenience, but ideally should be env var.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MODEL_NAME = "gemini-pro-latest"  # Fallback to pro-latest

SYSTEM_PROMPT = (
    "You are a precise stance classifier. Given a text, first assess if it is related to the Israel-Palestine topic. "
    "If related, assess sentiment/position. Return calibrated scores in [0,1] for pro_palestine and pro_israel (independent)."
)

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

def ensure_gemini_client():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    return genai.Client(api_key=GOOGLE_API_KEY)

def build_user_prompt(text: str, threshold: float) -> str:
    return (
        "Classify stance.") + (
        "\nInstructions:" 
        "\n- Output a single JSON object only, no prose."
        "\n- Fields: is_related (boolean), pro_palestine_score (0..1), pro_israel_score (0..1), rationale (short)."
        "\n- If the text is NOT related to the Israel-Palestine conflict, set is_related to false and scores to 0."
        "\n- DO NOT include a 'stance' field; it will be computed externally using the difference threshold."
        f"\n- Scores should reflect advocacy/support tone, not just mentions. Threshold for neutrality: {threshold:.3f}."
        "\n\nText:\n" + text.strip()
    )

def call_gemini_classify(
    client: Any,
    model: str,
    text: str,
    threshold: float,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Call Gemini model to get stance scores for a single text.
    Returns dict with scores and rationale.
    """
    last_err: Optional[Exception] = None
    
    # Configure generation to enforce JSON output if possible, 
    # or just rely on the prompt instructions.
    # Gemini 1.5+ supports response_mime_type="application/json"
    
    gen_config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ]
    )

    prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(text, threshold)}"

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=gen_config,
            )
            
            if not resp.text:
                raise ValueError("Empty response from Gemini")

            # Clean up potential markdown code blocks if the model adds them
            content = resp.text.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content.strip())
            
        except Exception as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
            
    raise RuntimeError(f"Gemini classification failed after {max_retries} attempts: {last_err}")

def compute_stance(pro_pal: float, pro_isr: float, neutral_delta: float, is_related: bool = True) -> str:
    if not is_related:
        return "Non-related"
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
    limit: Optional[int] = 10,
) -> pd.DataFrame:
    client = ensure_gemini_client()

    texts: List[str] = df[text_col].astype(str).tolist()
    if limit is not None:
        texts = texts[:limit]

    pro_pal_scores: List[Optional[float]] = []
    pro_isr_scores: List[Optional[float]] = []
    stances: List[Optional[str]] = []
    rationales: List[Optional[str]] = []

    print(f"Classifying {len(texts)} texts using Gemini ({model})...")

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
            result = call_gemini_classify(client, model, text_for_model, neutral_delta)
            is_related = bool(result.get("is_related", True))
            pro_pal = float(result.get("pro_palestine_score", 0.0))
            pro_isr = float(result.get("pro_israel_score", 0.0))
            stance = compute_stance(pro_pal, pro_isr, neutral_delta, is_related)
            rationale = str(result.get("rationale", "")).strip() or None
        except Exception as exc:
            print(f"Error on row {idx}: {exc}")
            pro_pal = None
            pro_isr = None
            stance = None
            rationale = f"Gemini error: {exc}"

        pro_pal_scores.append(pro_pal)
        pro_isr_scores.append(pro_isr)
        stances.append(stance)
        rationales.append(rationale)
        
        # Simple progress indicator
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(texts)}")

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

    parser = argparse.ArgumentParser(description="Classify stance using Google Gemini.")
    parser.add_argument("--input", default=os.path.join("Stance", "DataSample.csv"), help="Path to input CSV")
    parser.add_argument("--text-col", default="text", help="Name of the text column")
    parser.add_argument("--model", default=MODEL_NAME, help=f"Gemini model name (default: {MODEL_NAME})")
    parser.add_argument("--neutral-delta", type=float, default=0.2, help="Max score difference to be considered Neutral")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of rows")
    parser.add_argument("--max-chars", type=int, default=4000, help="Max chars per text")
    parser.add_argument("--output", default=os.path.join("Stance", "DataSample_stance_gemini.csv"), help="Output CSV path")

    args = parser.parse_args(argv)

    df = read_csv_robust(args.input)
    if args.text_col not in df.columns:
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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"Saved Gemini stance results to: {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
