import os
import sys
import re
from typing import List, Optional

import pandas as pd


def read_csv_robust(csv_path: str) -> pd.DataFrame:
    read_kwargs = dict(
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )
    try:
        return pd.read_csv(csv_path, engine="python", **read_kwargs)
    except Exception:
        return pd.read_csv(csv_path, engine="c", **read_kwargs)


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#(\w+)")
MENTION_RE = re.compile(r"@[\w.]+")
NON_WORD_RE = re.compile(r"[^\w\s#]+", re.UNICODE)
WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    t = text or ""
    t = t.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    # keep hashtags (we boost them later), strip other punctuation
    t = NON_WORD_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def tokenize(text: str) -> List[str]:
    return text.split()


PRO_PALESTINE_TERMS = [
    # core
    "freepalestine", "free_palestine", "free palestine", "gaza", "gazaunderattack",
    "palestine", "palestinian", "west bank", "rafah", "genocide", "occupation",
    "nakba", "idf crimes", "ceasefire now", "from the river to the sea",
    # hashtags/common tokens
    "#freepalestine", "#gazagenocide", "#gazaunderattack", "#palestine", "#gaza",
    # support words
    "stand with gaza", "stand with palestine", "save gaza", "stop the war",
]


PRO_ISRAEL_TERMS = [
    # core
    "standwithisrael", "stand_with_israel", "stand with israel", "israel", "idf",
    "bring them home", "hostages", "anti hamas", "destroy hamas", "self defense",
    "right to defend", "terrorists", "oct 7", "october 7",
    # hashtags/common tokens
    "#standwithisrael", "#istandwithisrael", "#bringthemhome", "#freehostages",
]


NEGATION_TOKENS = {"not", "no", "never", "without", "stop", "against"}


def count_matches(tokens: List[str], phrases: List[str]) -> float:
    """
    Score occurrences of phrases in token stream.
    - Unigram matches: direct token matches
    - N-gram phrases: sliding window exact match
    - Hashtag boost: +50% if matched token started with '#'
    - Negation: if a negation token appears within prev 3 tokens, reduce weight by 60%
    """
    score = 0.0
    text = " ".join(tokens)

    # Precompute positions for negation handling
    neg_positions = {i for i, tok in enumerate(tokens) if tok in NEGATION_TOKENS}

    def negation_near(idx: int) -> bool:
        return any((idx - k) in neg_positions for k in (1, 2, 3))

    # Build mapping for n-gram search
    for phrase in phrases:
        p = phrase.lower().strip()
        if not p:
            continue

        if " " in p:  # multi-token
            # simple exact match count on text; roughly token boundary safe by surrounding spaces
            occurrences = 0
            start = 0
            needle = " " + p + " "
            hay = " " + text + " "
            while True:
                pos = hay.find(needle, start)
                if pos == -1:
                    break
                occurrences += 1
                start = pos + len(needle)
            score += occurrences * 1.5
        else:
            for idx, tok in enumerate(tokens):
                if tok == p or tok.lstrip('#') == p:
                    w = 1.0
                    if tok.startswith('#'):
                        w *= 1.5
                    if negation_near(idx):
                        w *= 0.4
                    score += w
    return score


def normalize_pair(a: float, b: float) -> (float, float):
    s = a + b
    if s <= 0:
        return 0.0, 0.0
    return a / s, b / s


def classify_rules(text: str, neutral_delta: float) -> dict:
    norm = normalize_text(text)
    tokens = tokenize(norm)
    pro_pal_raw = count_matches(tokens, PRO_PALESTINE_TERMS)
    pro_isr_raw = count_matches(tokens, PRO_ISRAEL_TERMS)

    pro_pal, pro_isr = normalize_pair(pro_pal_raw, pro_isr_raw)
    diff = abs(pro_pal - pro_isr)
    if diff < neutral_delta:
        stance = "Neutral"
    else:
        stance = "Pro-Palestine" if pro_pal > pro_isr else "Pro-Israel"

    rationale = f"rule-based scores (normalized) pal={pro_pal:.3f}, isr={pro_isr:.3f}; diff={diff:.3f}"
    return {
        "pro_palestine_score": pro_pal,
        "pro_israel_score": pro_isr,
        "stance": stance,
        "stance_rationale": rationale,
    }


def classify_dataframe(
    df: pd.DataFrame,
    text_col: str,
    neutral_delta: float,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    out = df.copy()
    texts = out[text_col].astype(str).tolist()
    if limit is not None:
        texts = texts[:limit]

    pal_scores: List[Optional[float]] = []
    isr_scores: List[Optional[float]] = []
    stances: List[Optional[str]] = []
    rationales: List[Optional[str]] = []

    for t in texts:
        if not (t or "").strip():
            pal_scores.append(None)
            isr_scores.append(None)
            stances.append(None)
            rationales.append(None)
            continue
        res = classify_rules(t, neutral_delta)
        pal_scores.append(res["pro_palestine_score"])  # type: ignore
        isr_scores.append(res["pro_israel_score"])     # type: ignore
        stances.append(res["stance"])                  # type: ignore
        rationales.append(res["stance_rationale"])     # type: ignore

    if limit is not None:
        out = out.iloc[:limit].copy()

    out["pro_palestine_score"] = pal_scores
    out["pro_israel_score"] = isr_scores
    out["stance"] = stances
    out["stance_rationale"] = rationales
    return out


def main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Rule-based stance classifier (no LLM).")
    parser.add_argument("--input", default=os.path.join("Stance", "DataSample.csv"), help="Path to input CSV (default: Stance/DataSample.csv)")
    parser.add_argument("--text-col", default="text", help="Name of the text column (default: text)")
    parser.add_argument("--neutral-delta", type=float, default=0.2, help="Max score difference to be considered Neutral (default: 0.2)")
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit number of rows (debug)")
    parser.add_argument("--output", default=os.path.join("Stance", "DataSample_stance_rule.csv"), help="Output CSV path (default: Stance/DataSample_stance_rule.csv)")

    args = parser.parse_args(argv)

    df = read_csv_robust(args.input)
    if args.text_col not in df.columns:
        candidates = ["text", "post_text", "link_attachment.caption", "link_attachment.description"]
        found = next((c for c in candidates if c in df.columns), None)
        if not found:
            raise KeyError(f"Text column '{args.text_col}' not found. Available columns: {list(df.columns)}")
        args.text_col = found

    result_df = classify_dataframe(df, args.text_col, args.neutral_delta, args.limit)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"Saved rule-based stance results to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


