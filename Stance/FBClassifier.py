#!/usr/bin/env python3
"""
Classify Facebook posts using the trained Ministral-8B stance classifier.

This script loads the trained model from best_adapter/ and applies it to
a new CSV file containing Facebook posts. It uses the same 3-class system:
- Pro-Palestinian
- Pro-Israeli  
- Neutral (merged from Other, Off-topic, Anti-War_Pro-Peace)
"""

from __future__ import annotations

import os
import json
import logging
import math
import re
from pathlib import Path
from typing import List, Sequence
from concurrent.futures import ProcessPoolExecutor

import warnings
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig

# Optional: for language filtering
try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    logging.warning("langdetect not installed. Language filtering disabled.")

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────
MODEL_DIR = "./best_adapter"  # Path to trained model directory
TRAIN_CSV = "./Stance/gaza_stance_sampled_classified.csv"  # For label mapping fallback

# Default paths (can be overridden via command line)
DEFAULT_INPUT_CSV = "/home/jose/Documents/UNI/Mémoire/Data/Final data to use/finaldataSample20251127.csv"  # Set your input CSV path here
DEFAULT_OUTPUT_CSV = None  # Will be auto-generated if None

# Processing settings
BATCH_SIZE = 32
BATCH_SAVE_EVERY = 100  # Save every N batches
MAX_LENGTH = 512
SEP = " - "
CONSTRUCTED_COL = "constructed_text"
PRED_COL = "predicted_category"

# Language filtering (optional)
FILTER_ENGLISH = True
TARGET_LANG = "en"
NUM_WORDS_SAMPLE = 100
NUM_WORKERS = None  # None = use all CPU cores

# ──────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
log = logging.getLogger("stance_classifier")

# ──────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────

def strip_invisible(text: str) -> str:
    """Remove zero-width characters."""
    zero_width_re = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]")
    return zero_width_re.sub("", text)


def concatenate_fields(values: Sequence[str | float | None], *, sep: str = SEP) -> str:
    """Concatenate text fields, avoiding duplicates."""
    parts: List[str] = []
    for val in values:
        if not isinstance(val, str):
            continue
        val_clean = val.strip()
        if not val_clean:
            continue
        current = sep.join(parts).lower()
        if val_clean.lower() in current:
            continue
        parts.append(val_clean)
    return sep.join(parts)


def process_facebook(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct full text from Facebook CSV columns."""
    text_cols = ["Message", "Description", "Image Text", "Link Text"]
    df[CONSTRUCTED_COL] = df.apply(
        lambda row: concatenate_fields([row.get(c) for c in text_cols]), axis=1
    )
    return df


def safe_read_csv(path: str | Path) -> pd.DataFrame:
    """Robust CSV reader with fallback."""
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.ParserError as err:
        log.warning(f"Standard parser failed for {path}. Retrying with engine='python'...")
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


# ──────────────────────────────────────────────────────────────────────────
# LANGUAGE FILTERING (OPTIONAL)
# ──────────────────────────────────────────────────────────────────────────

def _detect_lang_worker(args):
    """Worker for parallel language detection."""
    idx, txt = args
    if not HAS_LANGDETECT:
        return idx, True  # Skip filtering if langdetect not available
    words = txt.split()
    sample = " ".join(words[:NUM_WORDS_SAMPLE])
    if not sample.strip():
        return idx, False
    try:
        return idx, detect(sample) == TARGET_LANG
    except (LangDetectException, Exception):
        return idx, False


def filter_english(texts: list[str], *, workers: int | None = NUM_WORKERS) -> list[bool]:
    """Filter texts to keep only English ones."""
    if not HAS_LANGDETECT:
        log.warning("langdetect not available. Skipping language filtering.")
        return [True] * len(texts)
    
    workers = workers or os.cpu_count() or 4
    log.info(f"Detecting language on {len(texts)} texts with {workers} workers...")
    
    flags = [False] * len(texts)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, ok in tqdm(
            ex.map(_detect_lang_worker, enumerate(texts), chunksize=512),
            total=len(texts),
            desc="Lang-detect",
            unit="post"
        ):
            flags[idx] = ok
    return flags


# ──────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────

def load_classifier(model_dir: str, *, train_csv: str = TRAIN_CSV):
    """Load the trained classifier model and tokenizer."""
    log.info(f"Loading model from {model_dir}...")
    
    # Load PEFT config
    p_cfg = PeftConfig.from_pretrained(model_dir)
    base_name = p_cfg.base_model_name_or_path
    
    # Determine label mapping
    id2label = getattr(p_cfg, "id2label", None)
    num_labels = None
    
    # Try to get from PEFT config
    if id2label:
        num_labels = len(id2label)
        # Convert to dict if needed
        if isinstance(id2label, list):
            id2label = {i: label for i, label in enumerate(id2label)}
    
    # If not found, try adapter_config.json
    if not id2label or not num_labels:
        try:
            with open(Path(model_dir) / "adapter_config.json") as f:
                raw = json.load(f)
            num_labels = raw.get("num_labels")
            id2label_dict = raw.get("id2label", {})
            if id2label_dict:
                id2label = {int(k): v for k, v in id2label_dict.items()}
                num_labels = len(id2label)
        except (FileNotFoundError, KeyError, ValueError):
            pass
    
    # Fallback: reconstruct from training CSV
    if not id2label or not num_labels:
        log.info(f"Reconstructing labels from {train_csv}...")
        try:
            df = pd.read_csv(train_csv)
            if "gpt_category" in df.columns:
                cat_col = "gpt_category"
            elif "cat" in df.columns:
                cat_col = "cat"
            else:
                raise ValueError("Cannot find category column in training CSV")
            
            # Apply same class merging as training
            FUSE_MAP = {
                "Pro-Palestinian": "Pro-Palestinian",
                "Pro-Israeli": "Pro-Israeli",
                "Other": "Neutral",
                "Off-topic": "Neutral",
                "Anti-War_Pro-Peace": "Neutral",
            }
            cats = sorted(df[cat_col].map(FUSE_MAP).dropna().unique())
            id2label = {i: c for i, c in enumerate(cats)}
            num_labels = len(id2label)
        except Exception as e:
            log.error(f"Failed to reconstruct labels: {e}")
            raise ValueError(f"Could not determine label mapping. Please check {train_csv} exists and has the correct columns.")
    
    # Validate we have labels
    if not id2label or num_labels is None or num_labels == 0:
        raise ValueError(f"Invalid label mapping: id2label={id2label}, num_labels={num_labels}")
    
    label2id = {v: k for k, v in id2label.items()}
    log.info(f"Label mapping: {id2label}")
    log.info(f"Number of labels: {num_labels}")
    
    # Load config - only pass num_labels and id2label if they're valid
    cfg = AutoConfig.from_pretrained(
        base_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    
    # Setup quantization (same as training)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ),
    )
    
    # Load base model
    log.info(f"Loading base model: {base_name}")
    base = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        config=cfg,
        device_map="auto",
        quantization_config=bnb_cfg,
        attn_implementation="flash_attention_2",
    )
    
    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    if tok.pad_token_id is None:
        tok.add_special_tokens({'pad_token': '<pad>'})
        tok.pad_token = '<pad>'
        base.resize_token_embeddings(len(tok))
    base.config.pad_token_id = tok.pad_token_id
    
    # Load PEFT adapter
    try:
        model = PeftModel.from_pretrained(base, model_dir)
    except RuntimeError as e:
        log.warning(f"LoRA head incompatible ({e}) → using ignore_mismatched_sizes=True")
        model = PeftModel.from_pretrained(base, model_dir, ignore_mismatched_sizes=True)
    
    model.eval()
    model.config.pad_token_id = tok.pad_token_id
    
    # Build prompt function (same as training)
    cats_str = ", ".join(id2label.values())
    def build_prompt(txt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert assistant. "
                    "Classify the following text into one of these "
                    f"categories: {cats_str}. "
                    "Respond with the category label only."
                ),
            },
            {"role": "user", "content": txt},
        ]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        ).strip()
    
    return tok, model, build_prompt, id2label


# ──────────────────────────────────────────────────────────────────────────
# BATCH PREDICTION
# ──────────────────────────────────────────────────────────────────────────

def incremental_predict(
    df: pd.DataFrame,
    tok,
    model,
    build_prompt,
    id2label: dict,
    *,
    text_col: str = CONSTRUCTED_COL,
    batch_size: int = BATCH_SIZE,
    save_every: int = BATCH_SAVE_EVERY,
    out_path: str | Path | None = None,
) -> None:
    """Predict categories for texts, saving incrementally."""
    device = next(model.parameters()).device
    
    # Find rows to process
    to_process = df.index[df[PRED_COL].isna() | (df[PRED_COL] == "")].tolist()
    if not to_process:
        log.info("No rows to categorize (already complete).")
        return
    
    total_batches = math.ceil(len(to_process) / batch_size)
    batch_counter = 0
    
    log.info(f"Processing {len(to_process)} rows in {total_batches} batches...")
    
    for i in tqdm(
        range(0, len(to_process), batch_size),
        desc="Batch-predict",
        total=total_batches,
        unit="batch",
    ):
        batch_idx = to_process[i : i + batch_size]
        batch_texts = df.loc[batch_idx, text_col].apply(strip_invisible).tolist()
        prompts = [build_prompt(t) for t in batch_texts]
        
        # Tokenize
        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=False,  # Important: same as training
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        
        # Predict
        with torch.no_grad():
            logits = model(**enc).logits
        ids = torch.argmax(logits, dim=1).tolist()
        
        # Map to labels
        labels = [id2label[i] for i in ids]
        df.loc[batch_idx, PRED_COL] = labels
        
        batch_counter += 1
        if out_path and batch_counter % save_every == 0:
            log.info(f"Interim save → {out_path}")
            df.to_csv(out_path, index=False)
    
    if out_path:
        log.info(f"Final save → {out_path}")
        df.to_csv(out_path, index=False)


# ──────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────

def main(input_csv: str, output_csv: str | None = None, filter_lang: bool = FILTER_ENGLISH, model_dir: str | None = None):
    """Main classification pipeline."""
    # Use provided model_dir or fall back to global
    if model_dir is None:
        model_dir = MODEL_DIR
    
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    
    if output_csv is None:
        output_csv = input_csv.parent / f"{input_csv.stem}_classified{input_csv.suffix}"
    else:
        output_csv = Path(output_csv)
    
    # Load or create dataframe
    if output_csv.exists():
        log.info(f"Found existing output ({output_csv}) – resume mode.")
        df = safe_read_csv(output_csv)
        if CONSTRUCTED_COL not in df.columns:
            raw = safe_read_csv(input_csv)
            df_texts = process_facebook(raw)[[CONSTRUCTED_COL]]
            df = df.join(df_texts)
    else:
        log.info(f"Loading raw CSV: {input_csv}")
        df = process_facebook(safe_read_csv(input_csv))
        df[PRED_COL] = pd.NA
    
    # Language filtering (if enabled)
    if filter_lang:
        mask_uncat = df[PRED_COL].isna() | (df[PRED_COL] == "")
        to_check = df.loc[mask_uncat, CONSTRUCTED_COL].tolist()
        if to_check:
            log.info(f"Language filtering ({len(to_check)} rows)...")
            flags = filter_english(to_check)
            df = df.loc[
                ~mask_uncat | pd.Series(flags, index=df.loc[mask_uncat].index)
            ].reset_index(drop=True)
    
    # Load model and classify - pass model_dir instead of using global
    tok, model, build_prompt, id2label = load_classifier(model_dir)
    incremental_predict(
        df,
        tok,
        model,
        build_prompt,
        id2label,
        batch_size=BATCH_SIZE,
        save_every=BATCH_SAVE_EVERY,
        out_path=output_csv,
    )
    
    log.info(f"Completed! Results saved to {output_csv}")
    log.info(f"Total rows: {len(df)}")
    
    # Print summary
    if PRED_COL in df.columns:
        counts = df[PRED_COL].value_counts()
        log.info("Classification summary:")
        for cat, count in counts.items():
            log.info(f"  {cat}: {count} ({count/len(df)*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    # Declare global at the start of the block
    
    parser = argparse.ArgumentParser(
        description="Classify Facebook posts using trained stance classifier"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        nargs="?",
        default=None,
        help="Path to input CSV file with Facebook posts",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output CSV (default: input_name_classified.csv)",
    )
    parser.add_argument(
        "--no-lang-filter",
        action="store_true",
        help="Disable English language filtering",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=MODEL_DIR,
        help=f"Path to model directory (default: {MODEL_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for inference (default: {BATCH_SIZE})",
    )
    
    args = parser.parse_args()
    
    # Update global MODEL_DIR if specified
    if args.model_dir != MODEL_DIR:
        MODEL_DIR = args.model_dir
    
    # Handle default input CSV if not provided
    if args.input_csv is None:
        if DEFAULT_INPUT_CSV:
            args.input_csv = DEFAULT_INPUT_CSV
        else:
            parser.error("input_csv is required (or set DEFAULT_INPUT_CSV)")
    
    main(
        input_csv=args.input_csv,
        output_csv=args.output,
        filter_lang=not args.no_lang_filter,
        model_dir=args.model_dir,  # Pass as parameter instead of modifying global
    )