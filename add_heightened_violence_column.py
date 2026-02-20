#!/usr/bin/env python3
"""
Add Heightened Violence Column to Dataset

This script reads the Gaza War Timeline and adds a binary column to the dataset
indicating whether each post was made during a period of heightened violence
(excluding ceasefire periods).
"""

import pandas as pd
import sys
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import re

def parse_timeline_date(period_str: str) -> Optional[Tuple[datetime, datetime]]:
    """
    Parse a date or date range from the timeline.
    Returns (start_date, end_date) tuple or None if parsing fails.
    """
    period_str = period_str.strip()
    
    if "–" in period_str or " - " in period_str:
        separator = "–" if "–" in period_str else " - "
        parts = period_str.split(separator)
        if len(parts) != 2:
            return None
        
        start_str = parts[0].strip()
        end_str = parts[1].strip()
        date_formats = ["%m/%d/%y", "%m/%d/%Y", "%d/%m/%y", "%d/%m/%Y"]
        
        start_date = None
        end_date = None
        
        for fmt in date_formats:
            try:
                start_date = datetime.strptime(start_str, fmt)
                break
            except ValueError:
                continue
        
        for fmt in date_formats:
            try:
                end_date = datetime.strptime(end_str, fmt)
                break
            except ValueError:
                continue
        
        if start_date and end_date:
            return (start_date, end_date)
    else:
        date_formats = ["%m/%d/%y", "%m/%d/%Y", "%d/%m/%y", "%d/%m/%Y"]
        
        for fmt in date_formats:
            try:
                date_val = datetime.strptime(period_str, fmt)
                start_date = date_val - timedelta(days=3)
                end_date = date_val + timedelta(days=3)
                return (start_date, end_date)
            except ValueError:
                continue
    
    return None

def read_csv_robust(csv_path: str, sep: str = ",") -> pd.DataFrame:
    """
    Read CSV robustly, handling various formats and encoding issues.
    """
    read_kwargs = dict(
        dtype=str,
        keep_default_na=False,
        na_values=[],
        sep=sep
    )
    try:
        return pd.read_csv(csv_path, engine="python", **read_kwargs)
    except Exception as e:
        print(f"Warning: Error reading with python engine: {e}")
        try:
            return pd.read_csv(csv_path, engine="c", quoting=3, **read_kwargs)
        except Exception as e2:
            print(f"Error reading CSV: {e2}")
            raise

def is_in_violence_period(date_val: datetime, violence_periods: List[Tuple[datetime, datetime]]) -> bool:
    """
    Check if a date falls within any of the violence periods.
    """
    if pd.isna(date_val) or date_val is None:
        return False
    
    for start_date, end_date in violence_periods:
        if start_date <= date_val <= end_date:
            return True
    
    return False

def main():
    timeline_path = "/home/jose/Documents/UNI/Mémoire/Data/Gaza War Timeline_ Operations & Discourse.csv"
    data_path = "/home/jose/Documents/UNI/Mémoire/Scripts/finaldata_with_sentiment.csv"
    output_path = "/home/jose/Documents/UNI/Mémoire/Scripts/finaldata_with_sentiment.csv"
    
    print("Reading Gaza War Timeline...")
    timeline_df = read_csv_robust(timeline_path, sep=";")
    print(f"Loaded {len(timeline_df)} timeline events")
    
    violence_periods = []
    for idx, row in timeline_df.iterrows():
        event = str(row.get("Major Military Operations / Events", ""))
        period_str = str(row.get("Period", ""))
        
        if "Ceasefire" in event or "Humanitarian Pause" in event:
            print(f"Skipping ceasefire period: {period_str}")
            continue
        
        parsed = parse_timeline_date(period_str)
        if parsed:
            start_date, end_date = parsed
            violence_periods.append((start_date, end_date))
            print(f"Added violence period: {period_str} ({start_date.date()} to {end_date.date()})")
        else:
            print(f"Warning: Could not parse period: {period_str}")
    
    print(f"\nTotal violence periods identified: {len(violence_periods)}")
    
    print(f"\nReading dataset: {data_path}")
    df = read_csv_robust(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]}...")
    
    date_col = "Post Created Date"
    if date_col not in df.columns:
        print(f"Error: Column '{date_col}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        return 1
    
    print("\nParsing post dates...")
    df["post_date"] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
    
    missing_dates = df["post_date"].isna().sum()
    if missing_dates > 0:
        print(f"Warning: {missing_dates} rows have missing or invalid dates")
    
    if "heightened_violence" in df.columns:
        print("Warning: 'heightened_violence' column already exists. Overwriting...")
    
    print("\nCreating heightened_violence column...")
    df["heightened_violence"] = df["post_date"].apply(
        lambda x: 1 if is_in_violence_period(x, violence_periods) else 0
    )
    
    df = df.drop(columns=["post_date"])
    
    total_posts = len(df)
    violence_posts = df["heightened_violence"].sum()
    violence_percentage = (violence_posts / total_posts) * 100 if total_posts > 0 else 0
    
    print(f"\nSummary:")
    print(f"  Total posts: {total_posts:,}")
    print(f"  Posts during heightened violence: {violence_posts:,} ({violence_percentage:.2f}%)")
    print(f"  Posts during normal periods: {total_posts - violence_posts:,} ({100 - violence_percentage:.2f}%)")
    
    print(f"\nSaving updated dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    
    print("Done! Column 'heightened_violence' has been added to the dataset.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

