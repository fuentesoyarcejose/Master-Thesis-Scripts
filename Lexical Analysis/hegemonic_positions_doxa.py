#!/usr/bin/env python3
"""
Zero-Shot Learning for Hegemonic Position Analysis with Doxa Framework

This script uses zero-shot classification with hypothesis templates to identify 
hegemonic positions, incorporating the Doxa framework:

DOXA = Stance + Morals + Reach + Sourcing

HEGEMONIC INDEX (HI) = (Neutral Stance) + (Authority/Loyalty Score) + 
                       (High Account Reach) + (Zero-Shot Hegemony Score)

DEFINITION OF HEGEMONIC:
Hegemonic positions suggest a narrative toward pro-Israel or pro-allies 
(western powers). Key indicators:
1. Direct quotations from hegemonic sources (Sourcing)
2. Neutral stance (Stance)
3. High Authority/Loyalty moral scores (Morals)
4. High account reach (Reach)
5. Passive voice that softens/blurs responsibility

Uses Hugging Face transformers with hypothesis templates for zero-shot classification.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import glob
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Error: transformers is required. Install with: pip install transformers torch")
    sys.exit(1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("Warning: tqdm not available. Progress bars will be disabled.")
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import re
    RE_AVAILABLE = True
except ImportError:
    RE_AVAILABLE = False

HYPOTHESIS_TEMPLATE = "This text represents a {} perspective by using official statistics, institutional sourcing, selective passive/active voice (passive for victims, active for enemies), command-style language, and framing that aligns with dominant Western/Israeli narratives."

CANDIDATE_LABELS = [
    "hegemonic-mask",
    "hegemonic-sword", 
    "hegemonic-shield",
    "counter-hegemonic",
    "ambivalent"
]

STANCE_COLUMN = 'predicted_category'
AUTHORITY_COLUMN = 'csv_Authority'
LOYALTY_COLUMN = 'csv_Loyalty'
REACH_COLUMNS = ['csv_Followers at Posting', 'csv_Likes at Posting', 'csv_Total Interactions']


def detect_gpu():
    """
    Detect available GPU and return device ID.
    
    Returns:
    --------
    device_id : int
        GPU device ID (0+) if available, -1 for CPU
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_id = 0
            gpu_name = torch.cuda.get_device_name(device_id)
            gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9
            print(f"  ✓ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
            return device_id
        else:
            print(f"  ⚠ No GPU detected, using CPU")
            return -1
    except ImportError:
        print(f"  ⚠ PyTorch not available, using CPU")
        return -1


def load_zero_shot_classifier(model_name: str = "facebook/bart-large-mnli", device: int = -1, auto_detect_gpu: bool = True):
    """
    Load a zero-shot classification pipeline with hypothesis template support.
    Optimized for GPU usage.
    
    Parameters:
    -----------
    model_name : str
        Hugging Face model name
    device : int
        Device to use (-1 for CPU, 0+ for GPU). If -1 and auto_detect_gpu=True, will auto-detect GPU.
    auto_detect_gpu : bool
        If True and device=-1, automatically detect and use GPU if available
    
    Returns:
    --------
    classifier : pipeline
        Zero-shot classification pipeline
    """
    if auto_detect_gpu and device == -1:
        device = detect_gpu()
    
    print(f"Loading zero-shot classifier: {model_name}")
    if device >= 0:
        print(f"  Using GPU device: {device}")
    else:
        print(f"  Using CPU")
    print("  This may take a few minutes on first run (downloading model)...")
    
    try:
        model_kwargs = {}
        if device >= 0:
            try:
                import torch
            except ImportError:
                pass
        
        classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
            model_kwargs=model_kwargs
        )
        
        if device >= 0:
            print(f"  ✓ Model loaded successfully on GPU")
        else:
            print(f"  ✓ Model loaded successfully on CPU")
        return classifier
    except Exception as e:
        print(f"  Error loading model: {e}")
        if device >= 0:
            print(f"  Falling back to CPU...")
            try:
                classifier = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=-1
                )
                print(f"  ✓ Model loaded successfully on CPU (fallback)")
                return classifier
            except Exception as e2:
                print(f"  Error loading on CPU: {e2}")
                raise
        else:
            print(f"  Trying alternative model: typeform/distilbert-base-uncased-mnli")
            try:
                classifier = pipeline(
                    "zero-shot-classification",
                    model="typeform/distilbert-base-uncased-mnli",
                    device=device
                )
                print(f"  ✓ Alternative model loaded successfully")
                return classifier
            except Exception as e2:
                print(f"  Error loading alternative model: {e2}")
                raise


def classify_with_hypothesis(
    classifier,
    text: str,
    candidate_labels: List[str],
    hypothesis_template: str,
    multi_label: bool = False
) -> Dict[str, Any]:
    """
    Classify text using zero-shot classification with hypothesis template.
    
    Parameters:
    -----------
    classifier : pipeline
        Zero-shot classification pipeline
    text : str
        Text to classify
    candidate_labels : List[str]
        List of candidate labels
    hypothesis_template : str
        Hypothesis template with {} placeholder
    multi_label : bool
        Whether to allow multiple labels
    
    Returns:
    --------
    result : dict
        Classification result with labels and scores
    """
    if not text or pd.isna(text) or len(str(text).strip()) == 0:
        return {
            'labels': [],
            'scores': [],
            'top_label': None,
            'top_score': 0.0
        }
    
    try:
        result = classifier(
            str(text),
            candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=multi_label
        )
        
        top_label = result['labels'][0] if result['labels'] else None
        top_score = result['scores'][0] if result['scores'] else 0.0
        
        return {
            'labels': result['labels'],
            'scores': result['scores'],
            'top_label': top_label,
            'top_score': top_score
        }
    except Exception as e:
        print(f"  Warning: Classification error: {e}")
        return {
            'labels': [],
            'scores': [],
            'top_label': None,
            'top_score': 0.0
        }


def calculate_stance_score(stance_value: Any) -> float:
    """
    Calculate stance score component for HI based on predicted_category.
    
    Stance scoring (based on predicted_category column):
    - Pro-Israeli (S = 1.0): Overtly aligned with the hegemon. High baseline for supporting dominant power structure.
    - Neutral (S = 0.5): Neither explicitly pro-hegemonic nor counter-hegemonic.
    - Pro-Palestinian (S = 0.0): Inherently counter-hegemonic.
    
    Parameters:
    -----------
    stance_value : Any
        Stance value from predicted_category column
    
    Returns:
    --------
    score : float
        Stance score: 1.0 (Pro-Israeli), 0.5 (Neutral), or 0.0 (Pro-Palestinian)
    """
    if pd.isna(stance_value):
        return 0.5  # Default to neutral if unknown
    
    stance_map = {
        'pro-israeli': 1.0,
        'neutral': 0.5,
        'pro-palestinian': 0.0
    }
    
    stance_str = str(stance_value).lower().strip()
    s_align = stance_map.get(stance_str, 0.5)  # Default to neutral if not found
    
    return s_align


def calculate_morals_score(row: pd.Series) -> float:
    r"""
    Calculate compositional moral score component for HI.
    Formula: $M_{comp} = \frac{Score_{Auth} + Score_{Loy}}{\sum Score_{All}}$
    
    The ratio of (Authority + Loyalty) scores relative to the total moral foundations.
    
    Parameters:
    -----------
    row : pd.Series
        Row from dataframe containing all moral foundation scores
    
    Returns:
    --------
    score : float
        Compositional moral score (0.0 to 1.0, higher = more hegemonic)
    """
    moral_columns = [
        'Care', 'Harm',
        'Fairness', 'Cheating',
        'Loyalty', 'Betrayal',
        'Authority', 'Subversion',
        'Purity', 'Degradation'
    ]
    
    all_scores = []
    auth_score = None
    loy_score = None
    
    for col in moral_columns:
        val = None
        for col_name in [col, f'csv_{col}']:
            if col_name in row.index:
                val = row[col_name]
                break
        
        try:
            score_val = float(val) if pd.notna(val) else 0.0
            all_scores.append(score_val)
            
            if col == 'Authority':
                auth_score = score_val
            elif col == 'Loyalty':
                loy_score = score_val
            except (ValueError, TypeError):
                all_scores.append(0.0)
    
    total_sum = sum(all_scores)
    
    if total_sum == 0.0:
        return 0.5
    
    if auth_score is None:
        auth_score = 0.0
    if loy_score is None:
        loy_score = 0.0
    
    compositional_score = (auth_score + loy_score) / total_sum
    
    return min(max(compositional_score, 0.0), 1.0)


def calculate_reach_score(row: pd.Series, reach_columns: List[str], max_log_reach: float = None) -> float:
    r"""
    Calculate normalized reach score component for HI.
    Formula: $R_{norm} = \frac{\log(Reach_i + 1)}{\max(\log(Reach))}$
    
    Log-scaled and normalized account reach (Van Dijk's "Discursive Access").
    
    Parameters:
    -----------
    row : pd.Series
        Row from dataframe
    reach_columns : List[str]
        List of column names for reach metrics
    max_log_reach : float, optional
        Maximum log(reach) across all rows. If None, uses a default normalization.
    
    Returns:
    --------
    score : float
        Normalized reach score (0.0 to 1.0, higher = more hegemonic)
    """
    reach_values = []
    for col in reach_columns:
        if col in row.index:
            val = row[col]
                    try:
                        if pd.notna(val):
                            if isinstance(val, str):
                                val_clean = str(val).replace(',', '').strip()
                                reach_val = float(val_clean) if val_clean else 0.0
                            else:
                                reach_val = float(val)
                            if reach_val > 0:
                                reach_values.append(reach_val)
                    except (ValueError, TypeError):
                        continue
    
    if not reach_values:
        return 0.0
    
    max_reach = max(reach_values)
    log_reach_i = np.log1p(max_reach)
    
    if max_log_reach is not None and max_log_reach > 0:
        normalized = log_reach_i / max_log_reach
    else:
        max_expected = 10000000
        normalized = log_reach_i / np.log1p(max_expected)
    
    return min(max(normalized, 0.0), 1.0)


def calculate_structural_authority_score(page_category: Any = None, page_admin_country: Any = None) -> float:
    r"""
    Calculate Structural Authority Score (A) based on Page Category and Page Admin Top Country.
    
    This score reflects the structural position of the source in the hegemonic hierarchy.
    Uses a comprehensive scoring system based on country alignment and institutional type.
    
    Reference table (core combinations):
    - US/Israel + News/Gov = 1.0 (Core Hegemonic Bloc)
    - UK/EU + News/Gov = 0.8 (Close allies, high institutional capital)
    - International + NGO/Non-Profit = 0.4 (Ambivalent; often uses Care/Harm but stays neutral)
    - Palestine + News/Gov = 0.3 (Institutional but structurally marginalized/counter-hegemonic)
    - Palestine + Grassroots/Individual = 0.0 (Purely counter-hegemonic / periphery)
    
    Expanded scoring criteria:
    1. Country Groups (hegemonic alignment):
       - Tier 1 (Core Hegemonic): US, Israel = 1.0 base
       - Tier 2 (Close Allies): UK, EU countries, Canada, Australia, NZ = 0.8 base
       - Tier 3 (Strategic Partners): Saudi Arabia, UAE, other Gulf states = 0.6 base
       - Tier 4 (Neutral/International): Most other countries = 0.4 base
       - Tier 5 (Counter-Hegemonic): Palestine = 0.2 base
       - Tier 6 (Opposition): Iran, Syria (if applicable) = 0.1 base
    
    2. Page Type Categories (institutional authority):
       - Government/Official: Highest authority (multiply by 1.0)
       - News/Media: High authority (multiply by 0.95)
       - Corporate/Business: Medium-high (multiply by 0.7)
       - Educational: Medium (multiply by 0.6)
       - NGO/Non-Profit: Medium-low (multiply by 0.5)
       - Religious: Medium-low (multiply by 0.4)
       - Grassroots/Individual: Low (multiply by 0.3)
       - Entertainment: Low (multiply by 0.2)
    
    Parameters:
    -----------
    page_category : Any, optional
        Page Category from CSV (e.g., NEWS_SITE, TV_CHANNEL, NONPROFIT)
    page_admin_country : Any, optional
        Page Admin Top Country from CSV (e.g., US, IL, UK, PS)
    
    Returns:
    --------
    score : float
        Structural Authority Score (0.0 to 1.0, higher = more hegemonic)
    """
    page_cat_str = ""
    if page_category is not None and pd.notna(page_category):
        page_cat_str = str(page_category).upper()
    
    country_str = ""
    if page_admin_country is not None and pd.notna(page_admin_country):
        country_str = str(page_admin_country).upper()
    
    tier1_countries = ['US', 'USA', 'UNITED STATES', 'IL', 'ISR', 'ISRAEL']
    is_tier1 = any(country in country_str for country in tier1_countries)
    tier2_countries = [
        # UK
        'UK', 'GB', 'GBR', 'UNITED KINGDOM', 'GREAT BRITAIN',
        # EU Core
        'FR', 'FRA', 'FRANCE', 'DE', 'DEU', 'GERMANY', 'IT', 'ITA', 'ITALY',
        'ES', 'ESP', 'SPAIN', 'NL', 'NLD', 'NETHERLANDS', 'BE', 'BEL', 'BELGIUM',
        # EU Other
        'AT', 'AUT', 'AUSTRIA', 'SE', 'SWE', 'SWEDEN', 'DK', 'DNK', 'DENMARK',
        'FI', 'FIN', 'FINLAND', 'NO', 'NOR', 'NORWAY', 'IE', 'IRL', 'IRELAND',
        'PT', 'PRT', 'PORTUGAL', 'GR', 'GRC', 'GREECE', 'PL', 'POL', 'POLAND',
        'CZ', 'CZE', 'CZECH', 'HU', 'HUN', 'HUNGARY', 'RO', 'ROU', 'ROMANIA',
        'BG', 'BGR', 'BULGARIA', 'HR', 'HRV', 'CROATIA', 'SK', 'SVK', 'SLOVAKIA',
        'SI', 'SVN', 'SLOVENIA', 'EE', 'EST', 'ESTONIA', 'LV', 'LVA', 'LATVIA',
        'LT', 'LTU', 'LITHUANIA', 'LU', 'LUX', 'LUXEMBOURG', 'MT', 'MLT', 'MALTA',
        'CY', 'CYP', 'CYPRUS',
        # CANZUS (Close Anglosphere allies)
        'CA', 'CAN', 'CANADA', 'AU', 'AUS', 'AUSTRALIA', 'NZ', 'NZL', 'NEW ZEALAND'
    ]
    is_tier2 = any(country in country_str for country in tier2_countries)
    
    tier3_countries = [
        'SA', 'SAU', 'SAUDI ARABIA', 'AE', 'ARE', 'UNITED ARAB EMIRATES', 'UAE',
        'KW', 'KWT', 'KUWAIT', 'QA', 'QAT', 'QATAR', 'BH', 'BHR', 'BAHRAIN',
        'OM', 'OMN', 'OMAN', 'JO', 'JOR', 'JORDAN', 'EG', 'EGY', 'EGYPT',
        'MA', 'MAR', 'MOROCCO', 'TN', 'TUN', 'TUNISIA'
    ]
    is_tier3 = any(country in country_str for country in tier3_countries)
    
    tier5_countries = ['PS', 'PSE', 'PALESTINE', 'PALESTINIAN', 'PALESTINIAN TERRITORY',
                      'PALESTINIAN TERRITORIES', 'WEST BANK', 'GAZA']
    is_tier5 = any(country in country_str for country in tier5_countries)
    
    tier6_countries = ['IR', 'IRN', 'IRAN', 'SY', 'SYR', 'SYRIA']
    is_tier6 = any(country in country_str for country in tier6_countries)
    
    is_tier4 = not (is_tier1 or is_tier2 or is_tier3 or is_tier5 or is_tier6)
    
    gov_categories = [
        'GOVERNMENT', 'GOVERNMENT_OFFICIAL', 'POLITICIAN', 'POLITICAL_PARTY',
        'PUBLIC_FIGURE', 'POLITICAL_FIGURE', 'DIPLOMAT', 'AMBASSADOR'
    ]
    is_gov = any(cat in page_cat_str for cat in gov_categories)
    
    news_categories = [
        'NEWS_SITE', 'TV_CHANNEL', 'NEWSPAPER', 'MAGAZINE', 'TOPIC_NEWSPAPER',
        'MEDIA_NEWS_COMPANY', 'MEDIA_COMPANY', 'BROADCASTER', 'RADIO_STATION',
        'JOURNALIST', 'REPORTER', 'NEWS_AGENCY', 'WIRE_SERVICE'
    ]
    is_news = any(cat in page_cat_str for cat in news_categories)
    
    corporate_categories = [
        'COMPANY', 'CORPORATION', 'BUSINESS', 'ENTERPRISE', 'BRAND',
        'CONSULTING', 'FINANCIAL_SERVICES', 'BANK', 'INVESTMENT'
    ]
    is_corporate = any(cat in page_cat_str for cat in corporate_categories)
    
    educational_categories = [
        'EDUCATION', 'UNIVERSITY', 'SCHOOL', 'COLLEGE', 'ACADEMIC',
        'RESEARCH', 'INSTITUTE', 'THINK_TANK', 'POLICY_INSTITUTE'
    ]
    is_educational = any(cat in page_cat_str for cat in educational_categories)
    
    ngo_categories = [
        'NONPROFIT', 'NON_PROFIT', 'NGO', 'ORG_GENERAL', 'ORGANIZATION',
        'CHARITY', 'FOUNDATION', 'ASSOCIATION', 'SOCIAL_ORGANIZATION',
        'HUMANITARIAN', 'AID_ORGANIZATION', 'CIVIL_SOCIETY'
    ]
    is_ngo = any(cat in page_cat_str for cat in ngo_categories)
    
    religious_categories = [
        'RELIGIOUS', 'CHURCH', 'MOSQUE', 'SYNAGOGUE', 'TEMPLE',
        'FAITH_BASED', 'RELIGIOUS_ORGANIZATION', 'SPIRITUAL'
    ]
    is_religious = any(cat in page_cat_str for cat in religious_categories)
    
    grassroots_categories = [
        'PERSON', 'INDIVIDUAL', 'PERSONAL_BLOG', 'BLOG', 'PERSONAL',
        'COMMUNITY', 'LOCAL_BUSINESS', 'SMALL_BUSINESS', 'ACTIVIST',
        'GRASSROOTS', 'CITIZEN', 'CIVILIAN'
    ]
    is_grassroots = any(cat in page_cat_str for cat in grassroots_categories)
    
    entertainment_categories = [
        'ENTERTAINMENT', 'MUSIC', 'MOVIE', 'FILM', 'TV_SHOW',
        'CELEBRITY', 'ARTIST', 'ACTOR', 'SINGER', 'COMEDIAN',
        'SPORTS', 'ATHLETE', 'SPORTS_TEAM'
    ]
    is_entertainment = any(cat in page_cat_str for cat in entertainment_categories)
    
    if is_tier1:
        base_score = 1.0
    elif is_tier2:
        base_score = 0.8
    elif is_tier3:
        base_score = 0.6
    elif is_tier5:
        base_score = 0.2
    elif is_tier6:
        base_score = 0.1
    else:
        base_score = 0.4
    
    if is_gov:
        multiplier = 1.0
    elif is_news:
        multiplier = 0.95
    elif is_corporate:
        multiplier = 0.7
    elif is_educational:
        multiplier = 0.6
    elif is_ngo:
        multiplier = 0.5
    elif is_religious:
        multiplier = 0.4
    elif is_grassroots:
        multiplier = 0.3
    elif is_entertainment:
        multiplier = 0.2
    else:
        multiplier = 0.5
    
    final_score = base_score * multiplier
    
    if is_tier1 and (is_gov or is_news):
        return 1.0
    
    if is_tier2 and (is_gov or is_news):
        return 0.8
    
    if is_tier5 and is_grassroots:
        return 0.0
    
    if is_tier5 and (is_gov or is_news):
        return 0.3
    
    if is_tier4 and is_ngo:
        return 0.4
    
    if is_tier5:
        if is_ngo:
            final_score = min(final_score, 0.2)
        elif not (is_gov or is_news):
            final_score = min(final_score, 0.1)
    
    if is_tier6:
        final_score = min(final_score, 0.15)
    
    if is_tier3 and (is_gov or is_news):
        final_score = min(final_score + 0.1, 1.0)
    
    final_score = max(0.0, min(1.0, final_score))
    
    return final_score


def detect_hegemonic_linguistic_patterns(text: str, stance_score: float = 0.5) -> Dict[str, Any]:
    """
    Detect hegemonic linguistic patterns that indicate institutional power framing.
    
    Patterns:
    1. Officialization of Data: Detailed statistical breakdowns (e.g., "1,139 Israelis... 766 civilians... 373 security personnel")
    2. Selective Passive Voice: Passive for victims ("were killed"), active for enemies ("breaching," "attacking")
    3. Command Style: Imperative/authoritative language ("must stop", "do not forget")
    4. The Shield: Rationalization of state violence ("right to defend", "self-defense", "human shields")
    5. The Sword: Punitive language ("restrict", "consequences", "sanction") and appeals to leaders
    
    Parameters:
    -----------
    text : str
        Text to analyze
    stance_score : float
        Stance score (0.0 to 1.0) - used to boost certain patterns when Pro-Israel
    
    Returns:
    --------
    patterns : dict
        Detected hegemonic linguistic patterns
    """
    if not text or pd.isna(text) or len(str(text).strip()) == 0:
        return {
            'has_official_stats': False,
            'has_selective_passive_voice': False,
            'has_command_style': False,
            'linguistic_hegemony_score': 0.0
        }
    
    text_str = str(text).lower()
    text_original = str(text)
    
    detailed_numbers = re.findall(r'\d+[,\d]*\s+(?:israelis?|civilians?|personnel|nationals?|victims?|casualties?|security\s+personnel|foreign\s+nationals?)', text_original, re.IGNORECASE)
    has_official_stats = len(detailed_numbers) >= 2
    
    passive_victim_patterns = [
        r'\b(?:were|was)\s+(?:killed|taken\s+captive|injured|wounded|murdered|executed)',
        r'\b(?:were|was)\s+(?:killed|taken|injured|wounded)',
    ]
    has_passive_victims = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in passive_victim_patterns)
    
    active_enemy_patterns = [
        r'\b(?:breaching|attacking|directed\s+at|targeting|launching)',
        r'\b(?:militants?|hamas|terrorists?)\s+(?:breaching|attacking|launching)',
    ]
    has_active_enemies = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in active_enemy_patterns)
    
    has_selective_passive_voice = has_passive_victims and has_active_enemies
    
    command_patterns = [
        r'\b(?:must|should|need\s+to|have\s+to)\s+(?:stop|end|cease|halt)',
        r'\bdo\s+not\s+(?:forget|ignore|overlook)',
        r'\b(?:look\s+at|see|observe)\s+what\s+(?:they|it)\s+(?:are|is)\s+doing',
        r'\b(?:the\s+)?protests?\s+must\s+stop',
    ]
    has_command_style = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in command_patterns)
    
    shield_patterns = [
        r'\bright\s+to\s+defend',
        r'\bself[-\s]?defense',
        r'\bhuman\s+shields?',
        r'\bhiding\s+behind',
        r'\bno\s+choice',
        r'\bsecurity\s+measures?',
        r'\bonly\s+democracy',
        r'\bcivilized\s+world',
    ]
    has_shield = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in shield_patterns)
    
    punitive_patterns = [
        r'\brestrict',
        r'\bconsequences',
        r'\bsuffer',
        r'\bsanction',
        r'\bboycott',
        r'\bsuspend',
    ]
    has_punitive_language = (
        stance_score > 0.6 and 
        any(re.search(pattern, text_str, re.IGNORECASE) for pattern in punitive_patterns)
    )
    
    leader_names = [
        r'\bnetanyahu\b',
        r'\bbibi\b',
        r'\bbiden\b',
        r'\bbarkat\b',
        r'\bsmotrich\b',
        r'\bben\s+gvir\b',
        r'\bgallant\b',
        r'\bherzog\b',
        r'\blapid\b',
        r'\bgantz\b',
        r'\bsunak\b',
        r'\bmacron\b',
        r'בנימין\s+נתניהו',  # Hebrew: Benjamin Netanyahu
        r'ניר\s+ברקת',  # Hebrew: Nir Barkat
    ]
    has_appeal_to_leaders = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in leader_names)
    
    linguistic_score = 0.0
    if has_official_stats:
        linguistic_score += 0.4
    if has_selective_passive_voice:
        linguistic_score += 0.4
    if has_command_style:
        if stance_score > 0.8:
            linguistic_score += 0.5
        else:
            linguistic_score += 0.2
    if has_shield:
        linguistic_score += 0.3
    if has_punitive_language:
        linguistic_score += 0.3
    if has_appeal_to_leaders:
        linguistic_score += 0.4
    
    linguistic_score = min(linguistic_score, 1.0)
    
    return {
        'has_official_stats': has_official_stats,
        'has_selective_passive_voice': has_selective_passive_voice,
        'has_command_style': has_command_style,
        'has_shield': has_shield,
        'has_punitive_language': has_punitive_language,
        'has_appeal_to_leaders': has_appeal_to_leaders,
        'linguistic_hegemony_score': linguistic_score
    }


def detect_sourcing_indicators(text: str, page_category: Any = None, page_name: Any = None, page_admin_country: Any = None) -> Dict[str, Any]:
    """
    Detect sourcing indicators (institutional sources).
    
    Parameters:
    -----------
    text : str
        Text to analyze
    page_category : Any, optional
        Page category from CSV (e.g., NEWS_SITE, TV_CHANNEL)
    
    Returns:
    --------
    indicators : dict
        Sourcing indicators
    """
    if not text or pd.isna(text) or len(str(text).strip()) == 0:
        return {
            'has_institutional_source': False,
            'has_israeli_source': False,
            'has_us_western_source': False,
            'has_major_news_outlet': False,
            'is_criticizing_source': False,
            'sourcing_score': 0.0
        }
    
    text_str = str(text).lower()
    text_original = str(text)
    
    denialist_patterns = [
        r'palestine\s+(?:doesn\'?t|does\s+not)\s+exist',
        r'palestine\s+belongs\s+to\s+(?:the\s+)?jews?',
        r'no\s+such\s+thing\s+as\s+palestine',
        r'hamas\'\s+(?:fake|false)\s+(?:casualty|death|number)',  # "Hamas' fake casualty numbers"
        r'hamas\s+(?:fake|false)\s+(?:casualty|death|number)',  # "Hamas fake casualty numbers"
        # Specific Palestinian identity delegitimization patterns
        r'palestinian\s+identity\s+(?:was\s+)?created',
        r'palestinian\s+flag\s+(?:was\s+)?invented',
        r'flag\s+(?:was\s+)?invented',
        r'belonged\s+to\s+jordan',
        r'there\s+are\s+no\s+palestinians',
        r'palestinians?\s+(?:don\'?t|do\s+not)\s+exist',
        r'palestinian\s+identity\s+(?:was\s+)?created',
        r'flag\s+(?:was\s+)?invented',
        r'belonged\s+to\s+jordan',
        r'there\s+are\s+no\s+palestinians',
    ]
    has_denialist_language = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in denialist_patterns)
    
    dehumanizing_patterns = [
        r'hamas\s+(?:barbarians?|animals?|monsters?|savages?)',
        r'(?:barbarians?|animals?|monsters?|savages?)\s+(?:that|who)\s+(?:attacked|killed)',
    ]
    has_dehumanizing_language = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in dehumanizing_patterns)
    
    terror_labeling_patterns = [
        r'terror\s+(?:group|organization|organization)',
        r'\bterrorists?\b',  # "terrorist" or "terrorists" - strong hegemonic indicator
        r'\b(?:hamas|palestinian|militant)\s+terrorists?',  # "Hamas terrorists", "Palestinian terrorists"
    ]
    has_terror_labeling = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in terror_labeling_patterns)
    
    genocide_against_palestinians = re.search(
        r'genocide\s+(?:on|of|against)\s+palestinians?',
        text_str, re.IGNORECASE
    )
    has_genocide_against_palestinians = bool(genocide_against_palestinians)
    
    genocidal_threat = re.search(r'genocidal\s+threat', text_str, re.IGNORECASE)
    has_genocidal_threat = bool(genocidal_threat)
    
    genocidal_apartheid_patterns = [
        r'genocidal[,\s]+apartheid\s+israeli\s+state',
        r'apartheid\s+israeli\s+state',
        r'genocidal.*israeli\s+state',
    ]
    has_genocidal_apartheid = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in genocidal_apartheid_patterns)
    
    solidarity_patterns = [
        r'help\s+(?:our\s+)?(?:brother|sister|brothers|sisters)\s+(?:of|in)\s+gaza',
        r'brother\s+of\s+gaza',
        r'solidarity\s+with.*gaza',
        r'solidarity\s+with.*palestin',
    ]
    has_solidarity_language = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in solidarity_patterns)
    
    humanitarian_trapped_patterns = [
        r'(?:hundreds?|thousands?|many|scores?)\s+(?:including\s+)?(?:babies?|children|women|civilians?)\s+(?:still\s+)?(?:trapped|stuck|unable\s+to\s+flee)',
        r'(?:babies?|children|women|civilians?)\s+(?:still\s+)?(?:trapped|stuck|unable\s+to\s+flee)',
        r'(?:flee|fled).*but.*(?:hundreds?|thousands?|many|babies?|children|women|civilians?).*trapped',
    ]
    has_humanitarian_trapped = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in humanitarian_trapped_patterns)
    
    criticism_patterns = [
        r'(?:went\s+livid|damaging\s+the\s+image|harming\s+the\s+narrative)',
        r'(?:prove\s+that|expose|reveal|leak)',
        r'(?:war\s+crime|killing\s+civilians)',
        r'(?:hypocrisy|hypocricy|hypocritical)',  # Criticism of hypocrisy
        r'(?:criticiz|criticis|disastrously\s+misplayed)',  # Criticism language
        r'(?:playing\s+politics)',
    ]
    is_criticizing = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in criticism_patterns)
    
    if has_genocide_against_palestinians or has_genocidal_threat or has_genocidal_apartheid:
        is_criticizing = True
    
    criticism_of_officials = re.search(
        r'(?:biden|president|administration)\s+(?:has\s+shown|is\s+playing|misplayed|disastrous)',
        text_str, re.IGNORECASE
    )
    if criticism_of_officials:
        is_criticizing = True
    
    generic_violence_patterns = [
        r'(?:attacks?\s+(?:in|on|spark))',  # "attacks in Rafah spark" - no attribution
        r'(?:clashes?\s+(?:occurred|broke\s+out|erupted))',  # "clashes occurred" - no attribution
        r'(?:violence\s+(?:erupted|broke\s+out))',  # "violence erupted" - no attribution
        r'(?:killed\s+in)',  # "killed in the hospital" - no attribution
        r'(?:died\s+due\s+to)',  # "died due to" - no attribution
    ]
    has_generic_violence = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in generic_violence_patterns)
    
    active_attribution_patterns = [
        r'(?:israeli\s+(?:sniper|soldier|forces?|military|idf|troops|army)\s+(?:kills?|killed|murdered|shot|shoots?|bombed|bombing|attacked|attacking))',  # "Israeli sniper kills"
        r'(?:israeli\s+(?:sniper|soldier|forces?|military|idf|troops|army)\s+(?:targeted|targeting|struck|striking))',  # "Israeli forces targeted"
        r'(?:israel\s+(?:killed|kills|murdered|attacked|bombed|struck))',  # "Israel killed"
        r'(?:israeli\s+(?:bombing|bombardment|strike|strikes|attack|attacks))',  # "Israeli bombing"
        r'(?:israeli\s+attack)',  # "Israeli attack" - counter-hegemonic
        r'(?:#israeli\s+aggression)',  # "#Israeli aggression" - counter-hegemonic
        r'(?:israeli\s+aggression)',  # "Israeli aggression" - counter-hegemonic
        r'(?:following\s+(?:an\s+)?israeli\s+(?:bombing|strike|attack))',  # "following an Israeli bombing"
        r'(?:israeli\s+(?:airstrike|airstrikes|shelling|artillery))',  # "Israeli airstrike"
        r'(?:idf\s+(?:killed|kills|attacked|bombed|struck|targeted))',  # "IDF killed"
        r'(?:israeli\s+forces?\s+(?:killed|kills|attacked|bombed|struck))',  # "Israeli forces killed"
        r'(?:israeli\s+army)',  # "Israeli army" - counter-hegemonic when used in critical context
        r'(?:israeli\s+army)',  # "Israeli army" - counter-hegemonic when used in critical context
        r'(?:israeli\s+occupation|israeli\s+occupation\s+forces)',  # "Israeli occupation"
        r'(?:israeli\s+(?:settlers?|settlement)\s+(?:attacked|killed|violence))',  # "Israeli settlers attacked"
        r'(?:exodus\s+of\s+.*\s+(?:from|due\s+to)\s+(?:israeli|israel))',  # "exodus of 360,000 from Israeli"
        r'(?:suffered\s+.*\s+(?:from|due\s+to)\s+(?:israeli|israel))',  # "suffered from Israeli"
        r'(?:assassination)',
    ]
    has_active_attribution = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in active_attribution_patterns)
    
    has_direct_action = has_active_attribution
    
    hegemonic_hashtags = [
        r'#hamasisisis',
        r'#freetheworldfromhamas',
        r'#israelwillwin',
        r'#hamasisterror',
        r'#destroyhamasterror',  # "DestroyHamasTerror"
        r'hamas-isis',
        r'hamas-isis\s+terrorists?',
    ]
    has_hegemonic_hashtag = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in hegemonic_hashtags)
    
    counter_hegemonic_hashtags = [
        r'#israelinewnazism',
        r'#gazastarving',
        r'#gazagenocide',
        r'#freepalestine',
        r'#gazaunderattack',
    ]
    has_counter_hegemonic_hashtag = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in counter_hegemonic_hashtags)
    
    free_palestine_patterns = [
        r'\bfree\s+palestine\b',
        r'\bfree\s+free\s+free\s+palestine\b',
        r'\bfree\s+palestin[ie]\b',  # "FREE PALESTINE" or "FREE PALESTINI"
    ]
    has_free_palestine = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in free_palestine_patterns)
    
    has_hebrew_script = bool(re.search(r'[\u0590-\u05FF]', text_original))
    
    hebrew_words = [
        r'\bshalom\b',  # "shalom" - peace/greeting
        r'\bchai\b',  # "chai" - life
        r'\bam\s+yisrael\s+chai\b',  # "Am Yisrael Chai" - The people of Israel live
        r'\bhashem\b',  # "Hashem" - God
        r'\btefillah\b',  # "tefillah" - prayer
        r'\btehillim\b',  # "tehillim" - psalms
    ]
    has_hebrew_words = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in hebrew_words)
    has_hebrew = has_hebrew_script or has_hebrew_words
    
    rumor_language = re.search(r'reportedly\s+(?:said|says|stated)', text_str, re.IGNORECASE)
    
    pro_war_patterns = [
        r'(?:brothers?|sisters?|soldiers?)\s+(?:fighting|fighting\s+in)\s+gaza',  # "brothers fighting in Gaza"
        r'(?:fighting|fighting\s+for)\s+(?:israel|our\s+country)',  # "fighting for Israel"
        r'(?:support|supporting)\s+(?:our\s+)?(?:troops|soldiers|military|idf)',  # "supporting our troops"
        r'(?:pray|praying)\s+(?:for|for\s+the)\s+(?:soldiers|troops|idf)',  # "praying for the soldiers"
    ]
    has_pro_war_language = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in pro_war_patterns)
    
    major_news_outlets = [
        r'\b(?:new\s+york\s+times|nytimes|nyt)\b',
        r'\b(?:washington\s+post|wapo)\b',
        r'\b(?:wall\s+street\s+journal|wsj)\b',
        r'\b(?:bbc|british\s+broadcasting)\b',
        r'\b(?:cnn|cable\s+news\s+network)\b',
        r'\b(?:fox\s+news)\b',
        r'\b(?:reuters)\b',
        r'\b(?:associated\s+press|ap\s+news)\b',
        r'\b(?:abc\s+news|abc12)\b',
        r'\b(?:cbs\s+news)\b',
        r'\b(?:nbc\s+news)\b',
    ]
    has_major_news = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in major_news_outlets)
    
    israeli_hegemonic_sources = [
        r'\b(?:times\s+of\s+israel|timesofisrael)\b',
        r'\bjpost\b',
        r'\bynet\b',
        r'\baipac\b',
        r'\bstandwithus\b',
        r'\bidf\b',
    ]
    has_israeli_hegemonic_source = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in israeli_hegemonic_sources)
    
    has_hegemonic_source = has_major_news or has_israeli_hegemonic_source
    
    religious_terms = [
        r'\brabbi\b',
        r'\btorah\b',
        r'\bsynagogue\b',
        r'\btefillah\b',  # prayer
        r'\btehillim\b',  # psalms
    ]
    military_terms = [
        r'\bsoldier\b',
        r'\bsoldiers\b',
        r'\bidf\b',
        r'\bwar\b',
        r'\barmy\b',
        r'\bmilitary\b',
    ]
    has_religious_term = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in religious_terms)
    has_military_term = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in military_terms)
    has_religious_authority_military = has_religious_term and has_military_term
    
    israeli_patterns = [
        r'\bisrael\s+(?:said|says|claims|stated|announced)',
        r'\bisraeli\s+(?:officials?|authorities?|government|prime\s+minister|military|idf)\s+(?:said|says|stated)',
        r'\baccording\s+to\s+israel',
        r'\b(?:idf|israeli\s+defense\s+forces?)\s+(?:said|says|stated)',
        r'\bisraeli\s+prime\s+minister',
        r'\bmember\s+of\s+the\s+israeli\s+prime\s+minister',
    ]
    
    us_western_patterns = [
        r'\b(?:white\s+house|whitehouse)\s+(?:said|says|stated)',
        r'\b(?:official|officials?)\s+(?:says|say|said|says)',
        r'\b(?:federal|federal\s+official)\s+(?:says|said)',
        r'\b(?:us|u\.s\.|united\s+states)\s+(?:officials?|government)\s+(?:said|says)',
        r'\b(?:hostage\s+coordinator|coordinator)\s+says',
        r'\b(?:law\s+enforcement|police)\s+(?:said|says|arrested)',
    ]
    
    us_politician_patterns = [
        r'\b(?:representative|representatives?|rep\.?)\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # "Representative Cori Bush"
        r'\b(?:senator|senators?|sen\.?)\s+[A-Z][a-z]+\s+[A-Z][a-z]+',  # "Senator X Y"
        r'\b(?:congressman|congresswoman|congressperson|member\s+of\s+congress)',
        r'\b(?:u\.s\.\s+house\s+of\s+representatives|house\s+of\s+representatives)',
        r'\b(?:u\.s\.\s+senate|us\s+senate)',
        r'\b(?:members?\s+of\s+congress)',
        r'\b(?:congress\s+(?:said|says|stated|member))',
        r'\b(?:biden|harris|president|vice\s+president)\s+(?:said|says|stated)',
        r'\b(?:blinken|secretary\s+of\s+state)\s+(?:said|says|stated)',
    ]
    has_us_politicians = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in us_politician_patterns)
    
    uk_official_patterns = [
        r'\b(?:rishi\s+sunak|uk\s+prime\s+minister|british\s+prime\s+minister)\s+(?:said|says|stated)',
        r'\b(?:uk|united\s+kingdom)\s+(?:official|officials?|government)\s+(?:said|says)',
        r'\b(?:british\s+government|british\s+official)\s+(?:said|says)',
    ]
    has_uk_officials = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in uk_official_patterns)
    
    irish_official_patterns = [
        r'\b(?:higgins|irish\s+president|ireland\s+president)\s+(?:said|says|stated)',
        r'\b(?:irish\s+government|ireland\s+government)\s+(?:said|says)',
    ]
    has_irish_officials = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in irish_official_patterns)
    
    iranian_official_patterns = [
        r'\b(?:iranian\s+president|iran\s+president|president\s+of\s+iran)\s+(?:said|says|stated)',
        r'\b(?:iranian\s+government|iran\s+government)\s+(?:said|says)',
        r'\b(?:iranian\s+official|iran\s+official)\s+(?:said|says)',
    ]
    has_iranian_officials = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in iranian_official_patterns)
    
    gaza_source_patterns = [
        r'\b(?:gaza\s+(?:source|sources|official|officials?|authorities?))\s+(?:said|says|stated)',
        r'\b(?:according\s+to\s+gaza)',
        r'\b(?:gaza\s+health\s+ministry)',
        r'\b(?:gaza\s+authorities?)',
    ]
    has_gaza_sources = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in gaza_source_patterns)
    
    al_jazeera_patterns = [
        r'\b(?:al\s+jazeera|aljazeera)\b',
        r'\b(?:jazeera)\s+(?:reports?|says?|stated?)',
    ]
    has_al_jazeera = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in al_jazeera_patterns)
    
    has_counter_hegemonic_page_name = False
    if page_name is not None and pd.notna(page_name):
        page_name_str = str(page_name).lower()
        counter_hegemonic_page_indicators = [
            'muslim voice',
            'palestinian',
            'palestine',
            'gaza',
            'peace and reconciliation',
        ]
        if any(indicator in page_name_str for indicator in counter_hegemonic_page_indicators):
            has_counter_hegemonic_page_name = True
    
    is_palestinian_source = False
    if page_admin_country is not None and pd.notna(page_admin_country):
        country_str = str(page_admin_country).upper()
        palestinian_countries = ['PS', 'PSE', 'PALESTINE', 'PALESTINIAN']
        if any(country in country_str for country in palestinian_countries):
            is_palestinian_source = True
    
    is_nonprofit_palestinian = False
    if page_category is not None and pd.notna(page_category):
        page_cat_str = str(page_category).upper()
        nonprofit_categories = ['NONPROFIT', 'NON_PROFIT', 'NGO', 'ORG_GENERAL']
        if any(cat in page_cat_str for cat in nonprofit_categories) and is_palestinian_source:
            is_nonprofit_palestinian = True
    
    has_israeli = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in israeli_patterns)
    has_us_western = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in us_western_patterns)
    if has_us_politicians or has_uk_officials:
        has_us_western = True
    if has_irish_officials and 'higgins' in text_str:
        is_criticizing = True
    
    is_institutional_media = False
    is_western_media = False
    if page_category is not None and pd.notna(page_category):
        page_cat_str = str(page_category).upper()
        institutional_categories = ['NEWS_SITE', 'TV_CHANNEL', 'NEWSPAPER', 'MAGAZINE', 'TOPIC_NEWSPAPER']
        is_institutional_media = any(cat in page_cat_str for cat in institutional_categories)
        western_categories = ['NEWS_SITE', 'TV_CHANNEL', 'NEWSPAPER', 'TOPIC_NEWSPAPER']
        is_western_media = any(cat in page_cat_str for cat in western_categories)
    
    has_institutional = (has_israeli or has_us_western or has_major_news or is_institutional_media or 
                        has_us_politicians or has_uk_officials) and not is_criticizing
    
    base_score = 0.0
    
    counter_hegemonic_indicators = [
        is_criticizing,
        has_genocide_against_palestinians,
        has_genocidal_threat,
        has_counter_hegemonic_page_name,
        has_counter_hegemonic_hashtag,
        has_iranian_officials,  # Iranian officials = enemy of Israel/hegemony
        has_gaza_sources,  # Gaza sources = Palestinian sources
        has_al_jazeera,  # Al Jazeera = counter-hegemonic source
        has_free_palestine,  # "FREE PALESTINE" = counter-hegemonic indicator
        is_palestinian_source,  # Palestinian-based source
        is_nonprofit_palestinian,  # Non-profit + Palestinian = strong counter-hegemonic
    ]
    
    if any(counter_hegemonic_indicators):
        base_score = 0.2
    elif has_direct_action and not has_institutional:
        base_score = 0.2
    elif has_institutional:
        base_score = 1.0
    elif is_institutional_media:
        if is_western_media:
            base_score = 0.7
        else:
            base_score = 0.4
    else:
        base_score = 0.0
    
    if base_score > 0.3:
        if has_denialist_language:
            base_score = min(1.0, base_score + 0.4)
        if has_dehumanizing_language:
            base_score = min(1.0, base_score + 0.35)
        if has_terror_labeling:
            base_score = min(1.0, base_score + 0.3)
        if has_hegemonic_hashtag:
            base_score = min(1.0, base_score + 0.3)
        if has_pro_war_language:
            base_score = min(1.0, base_score + 0.25)
        if has_hebrew:
            base_score = min(1.0, base_score + 0.2)
        if rumor_language:
            base_score = min(1.0, base_score + 0.2)
        if has_us_politicians and not is_criticizing:
            base_score = min(1.0, base_score + 0.25)
        if has_uk_officials and not is_criticizing:
            base_score = min(1.0, base_score + 0.25)
    
    if has_direct_action and base_score > 0.3:
        base_score = max(0.0, base_score - 0.2)
    
    if has_counter_hegemonic_hashtag and base_score > 0.3:
        base_score = max(0.0, base_score - 0.15)
    if has_al_jazeera and base_score > 0.3:
        base_score = max(0.0, base_score - 0.2)
    if has_free_palestine and base_score > 0.3:
        base_score = max(0.0, base_score - 0.2)
    if is_palestinian_source and base_score > 0.3:
        base_score = max(0.0, base_score - 0.15)
    if is_nonprofit_palestinian:
        base_score = 0.1
    
    sourcing_score = base_score
    
    return {
        'has_institutional_source': has_institutional,
        'has_israeli_source': has_israeli,
        'has_us_western_source': has_us_western,
        'has_major_news_outlet': has_major_news,
        'has_hegemonic_source': has_hegemonic_source,  # Major news or Israeli hegemonic sources
        'has_israeli_hegemonic_source': has_israeli_hegemonic_source,  # Israeli news outlets/pro-Israel orgs
        'has_religious_authority_military': has_religious_authority_military,  # Religious + military context
        'is_criticizing_source': is_criticizing,
        'has_direct_action': has_direct_action,  # Now equals has_active_attribution (explicit agency to hegemon)
        'has_active_attribution': has_active_attribution,  # Explicit attribution of agency to hegemon
        'has_generic_violence': has_generic_violence,  # Violence without attribution (not counted as resistance)
        'has_hegemonic_hashtag': has_hegemonic_hashtag,
        'has_hebrew': has_hebrew,
        'has_rumor_language': bool(rumor_language),
        'has_us_politicians': has_us_politicians,
        'has_uk_officials': has_uk_officials,
        'has_irish_officials': has_irish_officials,
        'has_denialist_language': has_denialist_language,
        'has_terror_labeling': has_terror_labeling,
        'has_genocide_against_palestinians': has_genocide_against_palestinians,
        'has_genocidal_threat': has_genocidal_threat,
        'has_genocidal_apartheid': has_genocidal_apartheid,
        'has_solidarity_language': has_solidarity_language,
        'has_humanitarian_trapped': has_humanitarian_trapped,
        'has_counter_hegemonic_page_name': has_counter_hegemonic_page_name,
        'has_counter_hegemonic_hashtag': has_counter_hegemonic_hashtag,
        'has_iranian_officials': has_iranian_officials,
        'has_gaza_sources': has_gaza_sources,
        'has_al_jazeera': has_al_jazeera,
        'has_free_palestine': has_free_palestine,
        'has_dehumanizing_language': has_dehumanizing_language,
        'has_pro_war_language': has_pro_war_language,
        'is_palestinian_source': is_palestinian_source,
        'is_nonprofit_palestinian': is_nonprofit_palestinian,
        'sourcing_score': sourcing_score
    }


def calculate_hegemonic_index(
    stance_score: float,
    morals_score: float,
    reach_score: float,
    zero_shot_score: float,
    zero_shot_label: str,
    authority_score: float = 0.0,
    linguistic_hegemony_score: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
    care_score: float = 0.0,
    harm_score: float = 0.0,
    resistance_indicators: Optional[Dict[str, bool]] = None,
    has_hegemonic_source: bool = False,
    has_religious_authority_military: bool = False,
    has_denialist_language: bool = False,
    has_terror_labeling: bool = False
) -> Dict[str, Any]:
    r"""
    Calculate Hegemonic Index (HI) using the specified formula.
    
    Formula: $HI_i = \omega_1(S_n) + \omega_2(M_{comp}) + \omega_3(R_{norm}) + \omega_4(Z_{text}) + \omega_5(A)$
    
    Where:
    - $S_n$: Stance score based on predicted_category (1.0 for Pro-Israeli, 0.5 for Neutral, 0.0 for Pro-Palestinian)
    - $M_{comp}$: Compositional Moral Score = (Authority + Loyalty) / sum(all moral foundations)
    - $R_{norm}$: Normalized Reach = log(Reach_i + 1) / max(log(Reach))
    - $Z_{text}$: Zero-Shot probability for "hegemonic" label
    - $A$: Structural Authority Score (based on Page Category and Page Admin Top Country)
    
    Parameters:
    -----------
    stance_score : float
        Stance score: 1.0 (Pro-Israeli), 0.5 (Neutral), or 0.0 (Pro-Palestinian) ($S_n$)
    morals_score : float
        Compositional moral score: (Auth + Loy) / sum(all) ($M_{comp}$)
    reach_score : float
        Normalized reach score ($R_{norm}$)
    zero_shot_score : float
        Zero-shot probability for "hegemonic" label ($Z_{text}$)
    zero_shot_label : str
        Zero-shot classification label
    authority_score : float
        Structural Authority Score based on Page Category and Page Admin Top Country ($A$)
    linguistic_hegemony_score : float, optional
        Score from linguistic pattern detection (official stats, selective passive voice, command style)
    weights : dict, optional
        Weights for each component. Default: ω₁=0.15, ω₂=0.25, ω₃=0.2, ω₄=0.2, ω₅=0.2
    care_score : float, optional
        Care moral foundation score (for resistance signal detection)
    harm_score : float, optional
        Harm moral foundation score (for resistance signal detection)
    resistance_indicators : dict, optional
        Dictionary of counter-hegemonic indicators (has_counter_hegemonic_hashtag, has_free_palestine, etc.)
        Used to determine if low HI score represents active resistance or just "noise"
    
    Returns:
    --------
    hi_data : dict
        Hegemonic Index components and final score
        
    Classification Logic:
    ---------------------
    Uses strict Resistance Condition ($C_{res}$):
    
    **Hegemony-Maintaining** (HI > 0.65):
        Discourse that actively naturalizes dominant power relations through high access,
        authority morals, and structural authority
    
    **Counter-Hegemonic** (HI ≤ 0.4 AND $C_{res}$ = True):
        Discourse that actively challenges power through moral critique, critical terminology,
        or political alignment. Requires positive resistance signals:
        - High Care/Harm scores (Care + Harm > 0.3), OR
        - Specific counter-hegemonic keywords/indicators, OR
        - Pro-Palestinian stance (stance_score == 0.0)
    
    **Ambivalent / Noise** (All others):
        Includes:
        - Transitional posts (0.4 < HI ≤ 0.65): Some hegemonic markers but mixed signals
        - Passive/Irrelevant posts (HI ≤ 0.4 AND $C_{res}$ = False): Low hegemony but lack
          active markers necessary to constitute dissent
        Principle: "Absence ≠ Resistance" - low hegemony alone doesn't mean counter-hegemonic
    """
    is_hegemonic_label = (
        "hegemonic-mask" in zero_shot_label.lower() or
        "hegemonic-sword" in zero_shot_label.lower() or
        "hegemonic-shield" in zero_shot_label.lower()
    )
    
    strong_counter_hegemonic_indicators = (
        resistance_indicators.get('has_genocidal_apartheid', False) or
        resistance_indicators.get('has_solidarity_language', False) or
        resistance_indicators.get('has_humanitarian_trapped', False) or
        resistance_indicators.get('has_genocide_against_palestinians', False) or
        resistance_indicators.get('has_direct_action', False) or
        resistance_indicators.get('has_free_palestine', False)
    )
    
    if is_hegemonic_label and not strong_counter_hegemonic_indicators:
        zero_shot_component = zero_shot_score
    elif zero_shot_label == "ambivalent" and linguistic_hegemony_score > 0.5 and not strong_counter_hegemonic_indicators:
        zero_shot_component = max(linguistic_hegemony_score, zero_shot_score * 0.7)
    elif strong_counter_hegemonic_indicators:
        zero_shot_component = 0.0
    else:
        zero_shot_component = 0.0
    
    if weights is None:
        weights = {
            'stance': 0.15,
            'morals': 0.25,
            'reach': 0.2,
            'zero_shot': 0.2,
            'authority': 0.2
        }
    
    hi_score = (
        weights['stance'] * stance_score +
        weights['morals'] * morals_score +
        weights['reach'] * reach_score +
        weights['zero_shot'] * zero_shot_component +
        weights['authority'] * authority_score
    )
    
    hegemonic_override_applied = False
    override_reason = None
    
    if linguistic_hegemony_score > 0.8:
        if hi_score < 0.75:
            hi_score = max(hi_score, 0.75)
            hegemonic_override_applied = True
            override_reason = "strong_linguistic_hegemony"
    
    if has_denialist_language:
        if hi_score < 0.85:
            hi_score = max(hi_score, 0.85)
            hegemonic_override_applied = True
            if override_reason is None:
                override_reason = "denialist_language"
    
    if has_terror_labeling:
        if hi_score < 0.7:
            hi_score = max(hi_score, 0.7)
            hegemonic_override_applied = True
            if override_reason is None:
                override_reason = "terrorist_labeling"
    
    if has_hegemonic_source or has_religious_authority_military:
        if hi_score < 0.75:
            hi_score = max(hi_score, 0.75)
            hegemonic_override_applied = True
            if override_reason is None:
                override_reason = "hegemonic_source_or_religious_authority"
    
    if hegemonic_override_applied:
        if "linguistic_hegemony" in override_reason or "denialist" in override_reason:
            if "hegemonic" not in zero_shot_label.lower():
                zero_shot_label = "hegemonic-shield"
        elif "terrorist_labeling" in override_reason:
            if "hegemonic" not in zero_shot_label.lower():
                zero_shot_label = "hegemonic-sword"
        elif "hegemonic_source" in override_reason:
            if "hegemonic" not in zero_shot_label.lower():
                zero_shot_label = "hegemonic-mask"
        
        zero_shot_score = hi_score
        zero_shot_component = zero_shot_score
    
    if resistance_indicators is None:
        resistance_indicators = {}
    
    resistance_signal = False
    
    care_harm_sum = care_score + harm_score
    if care_harm_sum > 0.3:
        resistance_signal = True
    
    active_resistance_markers = [
        resistance_indicators.get('has_counter_hegemonic_hashtag', False),  # #FreePalestine, #GazaGenocide, etc.
        resistance_indicators.get('has_free_palestine', False),  # "FREE PALESTINE" text
        resistance_indicators.get('has_al_jazeera', False),  # Al Jazeera source (counter-hegemonic media)
        resistance_indicators.get('has_direct_action', False),  # Active attribution (e.g., "Israeli forces killed")
        resistance_indicators.get('has_gaza_sources', False),  # Gaza/Palestinian sources
        resistance_indicators.get('is_palestinian_source', False),  # Palestinian-based source
        resistance_indicators.get('has_genocide_against_palestinians', False),  # Genocide accusations
        resistance_indicators.get('has_genocidal_apartheid', False),  # "genocidal, apartheid Israeli state"
        resistance_indicators.get('has_solidarity_language', False),  # "help our brother of Gaza"
        resistance_indicators.get('has_humanitarian_trapped', False),  # "babies still trapped"
        resistance_indicators.get('is_criticizing_source', False),  # Criticizing hegemonic sources/officials
    ]
    
    if any(active_resistance_markers):
        resistance_signal = True
    
    if stance_score == 0.0:
        resistance_signal = True
    
    if hi_score > 0.65:
        hi_category = "Hegemonic"
    elif hi_score <= 0.4:
        if resistance_signal:
            hi_category = "Counter-hegemonic"
        else:
            hi_category = "Ambivalent"
    else:
        hi_category = "Ambivalent"
    
    return {
            'hi_score': hi_score,
            'hi_category': hi_category,
            'stance_component': stance_score,  # $S_n$: 1.0 (Pro-Israeli), 0.5 (Neutral), 0.0 (Pro-Palestinian)
            'morals_component': morals_score,  # $M_{comp}$: Compositional Moral Score
            'reach_component': reach_score,   # $R_{norm}$: Normalized Reach
            'zero_shot_component': zero_shot_component,  # $Z_{text}$: Zero-Shot Hegemony Score
            'authority_component': authority_score,  # $A$: Structural Authority Score
            'resistance_signal': resistance_signal,  # $C_{res}$: Resistance Condition (True if positive signals present)
            'care_harm_sum': care_harm_sum,  # Care + Harm scores (for resistance signal detection)
            'hegemonic_override_applied': hegemonic_override_applied,  # True if override was applied
            'override_reason': override_reason,  # Reason for override (denialist_language, hegemonic_source_or_religious_authority)
            'zero_shot_label': zero_shot_label,  # Updated to "hegemonic" if override applied
            'zero_shot_score': zero_shot_score  # Updated to match override score if override applied
        }


def analyze_hegemonic_positions_doxa(
    df: pd.DataFrame,
    text_column: str = 'constructed_text',
    stance_column: str = 'predicted_category',
    authority_column: Optional[str] = None,
    loyalty_column: Optional[str] = None,
    reach_columns: Optional[List[str]] = None,
    model_name: str = "facebook/bart-large-mnli",
    batch_size: int = 32,
    max_texts: Optional[int] = None,
    device: int = -1
) -> pd.DataFrame:
    r"""
    Analyze hegemonic positions using Doxa framework and calculate HI.
    
    Formula: $HI_i = \omega_1(S_n) + \omega_2(M_{comp}) + \omega_3(R_{norm}) + \omega_4(Z_{text}) + \omega_5(A)$
    
    Where:
    - $S_n$: Stance score based on predicted_category (1.0 for Pro-Israeli, 0.5 for Neutral, 0.0 for Pro-Palestinian)
    - $M_{comp}$: Compositional Moral Score = (Authority + Loyalty) / sum(all moral foundations)
    - $R_{norm}$: Normalized Reach = log(Reach_i + 1) / max(log(Reach))
    - $Z_{text}$: Zero-Shot probability for "hegemonic" label
    - $A$: Structural Authority Score (based on Page Category and Page Admin Top Country)
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    text_column : str
        Name of column containing texts
    stance_column : str
        Name of column containing stance (predicted_category)
    authority_column : str, optional
        Name of column containing Authority moral score
    loyalty_column : str, optional
        Name of column containing Loyalty moral score
    reach_columns : List[str], optional
        List of column names for reach metrics
    model_name : str
        Hugging Face model name
    batch_size : int
        Batch size for processing
    max_texts : int, optional
        Maximum number of texts to process
    device : int
        Device to use (-1 for CPU, 0+ for GPU)
    
    Returns:
    --------
    results_df : DataFrame
        DataFrame with HI scores and components
    """
    print("="*70)
    print("Hegemonic Position Analysis - Doxa Framework")
    print("="*70)
    print("\nFormula: HI = ω₁(S_n) + ω₂(M_comp) + ω₃(R_norm) + ω₄(Z_text) + ω₅(A)")
    print("  Weights: ω₁ = 0.15 (Stance), ω₂ = 0.25 (Moral Composition),")
    print("           ω₃ = 0.2 (Structural Access/Reach), ω₄ = 0.2 (Linguistic Analysis),")
    print("           ω₅ = 0.2 (Structural Authority)")
    print("\n  Components:")
    print("    S_n: Stance score (1.0 for Pro-Israeli, 0.5 for Neutral, 0.0 for Pro-Palestinian)")
    print("    M_comp: (Authority + Loyalty) / sum(all moral foundations)")
    print("    R_norm: log(Reach_i + 1) / max(log(Reach))")
    print("    Z_text: Zero-Shot probability for 'hegemonic' label")
    print("    A: Structural Authority Score (based on Page Category and Page Admin Top Country)")
    print("\n  Classification:")
    print("    Hegemony-Maintaining (HI > 0.65): Naturalizes dominant power relations through")
    print("                                       high access, authority morals, structural authority")
    print("    Counter-Hegemonic (HI ≤ 0.4 AND C_res = True): Actively challenges power through")
    print("                                                    moral critique, critical terminology, or alignment")
    print("    Ambivalent / Noise (All others): Transitional (0.4 < HI ≤ 0.65) or")
    print("                                      Passive/Irrelevant (HI ≤ 0.4 AND C_res = False)")
    print("="*70)
    
    if authority_column is None:
        authority_column = AUTHORITY_COLUMN
    if loyalty_column is None:
        loyalty_column = LOYALTY_COLUMN
    if reach_columns is None:
        reach_columns = REACH_COLUMNS
    
    print(f"\nPreparing texts from column: '{text_column}'")
    texts = df[text_column].fillna("").astype(str).tolist()
    
    if max_texts is not None:
        texts = texts[:max_texts]
        df = df.iloc[:max_texts].copy()
        print(f"  Limited to {max_texts} texts for processing")
    
    print(f"  Total texts to process: {len(texts)}")
    
    classifier = load_zero_shot_classifier(model_name=model_name, device=device)
    
    print(f"\nClassifying texts with hypothesis template...")
    print(f"  Template: {HYPOTHESIS_TEMPLATE}")
    print(f"  Labels: {CANDIDATE_LABELS}")
    
    classification_results = []
    
    if device >= 0 and batch_size > 1:
        print(f"  Using batch processing (batch_size={batch_size}) for GPU efficiency...")
        for i in tqdm(range(0, len(texts), batch_size), desc="Classifying (batched)"):
            batch_texts = texts[i:i+batch_size]
            batch_results = []
            for text in batch_texts:
                result = classify_with_hypothesis(
                    classifier,
                    text,
                    CANDIDATE_LABELS,
                    HYPOTHESIS_TEMPLATE,
                    multi_label=False
                )
                batch_results.append(result)
            classification_results.extend(batch_results)
            
            if (i // batch_size) % 10 == 0 and device >= 0:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
    else:
        for i in tqdm(range(len(texts)), desc="Classifying"):
            result = classify_with_hypothesis(
                classifier,
                texts[i],
                CANDIDATE_LABELS,
                HYPOTHESIS_TEMPLATE,
                multi_label=False
            )
                except:
                    pass
    else:
        for i in tqdm(range(len(texts)), desc="Classifying"):
            result = classify_with_hypothesis(
                classifier,
                texts[i],
                CANDIDATE_LABELS,
                HYPOTHESIS_TEMPLATE,
                multi_label=False
            )
            classification_results.append(result)
    
    print(f"\nPre-calculating max(log(reach)) for normalization...")
    reach_cols_to_check = []
    for col in reach_columns:
        if col in df.columns:
            reach_cols_to_check.append(col)
        elif f'csv_{col}' in df.columns:
            reach_cols_to_check.append(f'csv_{col}')
    
    max_log_reach = 0.0
    if reach_cols_to_check:
        for col in reach_cols_to_check:
            reach_vals = df[col].fillna('0').astype(str).str.replace(',', '', regex=False)
            reach_vals = pd.to_numeric(reach_vals, errors='coerce').fillna(0)
            reach_vals = reach_vals[reach_vals > 0]
            if len(reach_vals) > 0:
                log_reaches = np.log1p(reach_vals)
                max_log_reach = max(max_log_reach, log_reaches.max())
    
    if max_log_reach == 0.0:
        max_log_reach = np.log1p(10000000)
        print(f"  Using default max_log_reach: {max_log_reach:.2f}")
    else:
        print(f"  Calculated max_log_reach: {max_log_reach:.2f}")
    
    print(f"\nCalculating Hegemonic Index (HI) for each text...")
    print(f"  Formula: HI = ω₁(S_n) + ω₂(M_comp) + ω₃(R_norm) + ω₄(Z_text) + ω₅(A)")
    print(f"  Where: S_n = Stance (1.0 Pro-Israeli, 0.5 Neutral, 0.0 Pro-Palestinian), M_comp = (Auth+Loy)/sum(all),")
    print(f"         R_norm = log(Reach+1)/max(log(Reach)), Z_text = P(hegemonic), A = Structural Authority Score")
    results_data = []
    
    for idx, result in enumerate(classification_results):
        if idx >= len(df):
            break
        
        row = df.iloc[idx]
        text = texts[idx] if idx < len(texts) else ""
        
        top_label = result.get('top_label', '')
        top_score = result.get('top_score', 0.0)
        
        counter_heg_score = 0.0
        for i, label in enumerate(result.get('labels', [])):
            if 'counter-hegemonic' in label.lower():
                counter_heg_score = result.get('scores', [])[i]
                break
        
        hegemonic_keywords = ["hegemonic-mask", "hegemonic-sword", "hegemonic-shield"]
        is_top_hegemonic = any(keyword in top_label.lower() for keyword in hegemonic_keywords)
        
        hegemonic_scores = []
        hegemonic_label_names = []
        for i, label in enumerate(result.get('labels', [])):
            label_lower = label.lower()
            for keyword in hegemonic_keywords:
                if keyword in label_lower:
                    hegemonic_scores.append(result.get('scores', [])[i])
                    hegemonic_label_names.append(label)
                    break
        
        max_heg_score = max(hegemonic_scores) if hegemonic_scores else 0.0
        
        if counter_heg_score >= 0.15 and max_heg_score < 0.35:
            zero_shot_label = "counter-hegemonic"
            zero_shot_score = counter_heg_score
        elif is_top_hegemonic:
            zero_shot_label = top_label
            zero_shot_score = top_score
        elif hegemonic_scores:
            max_heg_idx = hegemonic_scores.index(max_heg_score)
            zero_shot_label = hegemonic_label_names[max_heg_idx]
            zero_shot_score = max_heg_score
        else:
            zero_shot_label = top_label if top_label else "ambivalent"
            zero_shot_score = top_score
        
        stance_value = None
        for col_name in [stance_column, f'csv_{stance_column}']:
            if col_name in row.index:
                stance_value = row[col_name]
                break
        
        stance_score = calculate_stance_score(stance_value)
        morals_score = calculate_morals_score(row)
        
        reach_cols_to_use = []
        for col in reach_columns:
            if col in row.index:
                reach_cols_to_use.append(col)
            elif f'csv_{col}' in row.index:
                reach_cols_to_use.append(f'csv_{col}')
        
        reach_score = calculate_reach_score(row, reach_cols_to_use, max_log_reach=max_log_reach)
        
        page_category = None
        for col_name in ['Page Category', 'csv_Page Category']:
            if col_name in row.index:
                page_category = row[col_name]
                break
        
        page_name = None
        for col_name in ['User Name', 'csv_User Name', 'Page Name', 'csv_Page Name']:
            if col_name in row.index:
                page_name = row[col_name]
                break
        
        page_admin_country = None
        for col_name in ['Page Admin Top Country', 'csv_Page Admin Top Country']:
            if col_name in row.index:
                page_admin_country = row[col_name]
                break
        
        authority_score = calculate_structural_authority_score(
            page_category=page_category,
            page_admin_country=page_admin_country
        )
        
        sourcing_indicators = detect_sourcing_indicators(
            text, 
            page_category=page_category, 
            page_name=page_name,
            page_admin_country=page_admin_country
        )
        sourcing_score = sourcing_indicators['sourcing_score']
        
        linguistic_patterns = detect_hegemonic_linguistic_patterns(text, stance_score=stance_score)
        linguistic_hegemony_score = linguistic_patterns['linguistic_hegemony_score']
        
        care_score = 0.0
        harm_score = 0.0
        for col_name in ['Care', 'csv_Care']:
            if col_name in row.index:
                try:
                    care_score = float(row[col_name]) if pd.notna(row[col_name]) else 0.0
                    break
                except (ValueError, TypeError):
                    care_score = 0.0
        
        for col_name in ['Harm', 'csv_Harm']:
            if col_name in row.index:
                try:
                    harm_score = float(row[col_name]) if pd.notna(row[col_name]) else 0.0
                    break
                except (ValueError, TypeError):
                    harm_score = 0.0
        
        resistance_indicators = {
            'has_counter_hegemonic_hashtag': sourcing_indicators.get('has_counter_hegemonic_hashtag', False),
            'has_free_palestine': sourcing_indicators.get('has_free_palestine', False),
            'has_al_jazeera': sourcing_indicators.get('has_al_jazeera', False),
            'has_direct_action': sourcing_indicators.get('has_direct_action', False),
            'has_gaza_sources': sourcing_indicators.get('has_gaza_sources', False),
            'is_palestinian_source': sourcing_indicators.get('is_palestinian_source', False),
            'has_genocide_against_palestinians': sourcing_indicators.get('has_genocide_against_palestinians', False),
            'is_criticizing_source': sourcing_indicators.get('is_criticizing_source', False),
        }
        
        hi_data = calculate_hegemonic_index(
            stance_score,
            morals_score,
            reach_score,
            zero_shot_score,
            zero_shot_label,
            authority_score=authority_score,
            linguistic_hegemony_score=linguistic_hegemony_score,
            care_score=care_score,
            harm_score=harm_score,
            resistance_indicators=resistance_indicators,
            has_hegemonic_source=sourcing_indicators.get('has_hegemonic_source', False),
            has_religious_authority_military=sourcing_indicators.get('has_religious_authority_military', False),
            has_denialist_language=sourcing_indicators.get('has_denialist_language', False),
            has_terror_labeling=sourcing_indicators.get('has_terror_labeling', False)
        )
        
        # Compile results
        # Use zero_shot_label and zero_shot_score from hi_data (may have been overridden)
        final_zero_shot_label = hi_data.get('zero_shot_label', zero_shot_label)
        final_zero_shot_score = hi_data.get('zero_shot_score', zero_shot_score)
        
        row_data = {
            'text_index': idx,
            'text': text[:500] if text else "",  # Truncate for CSV
            'zero_shot_label': final_zero_shot_label,  # May have been overridden to "hegemonic"
            'zero_shot_score': final_zero_shot_score,  # May have been overridden to match HI score
            'hi_score': hi_data['hi_score'],
            'hi_category': hi_data['hi_category'],
            'stance_component': hi_data['stance_component'],  # $S_n$: 1.0 (Pro-Israeli), 0.5 (Neutral), 0.0 (Pro-Palestinian)
            'morals_component': hi_data['morals_component'],  # $M_{comp}$
            'reach_component': hi_data['reach_component'],   # $R_{norm}$
            'zero_shot_component': hi_data['zero_shot_component'],  # $Z_{text}$
            'authority_component': hi_data['authority_component'],  # $A$: Structural Authority Score
            'resistance_signal': hi_data.get('resistance_signal', False),  # $C_{res}$: Resistance Condition
            'care_harm_sum': hi_data.get('care_harm_sum', 0.0),  # Care + Harm scores (for resistance detection)
            'hegemonic_override_applied': hi_data.get('hegemonic_override_applied', False),  # True if override was applied
            'override_reason': hi_data.get('override_reason', None),  # Reason for override
            'has_institutional_source': sourcing_indicators['has_institutional_source'],
            'has_israeli_source': sourcing_indicators['has_israeli_source'],
            'has_us_western_source': sourcing_indicators['has_us_western_source'],
            'has_major_news_outlet': sourcing_indicators.get('has_major_news_outlet', False),
            'has_hegemonic_source': sourcing_indicators.get('has_hegemonic_source', False),  # Major news or Israeli hegemonic sources
            'has_israeli_hegemonic_source': sourcing_indicators.get('has_israeli_hegemonic_source', False),  # Israeli news/pro-Israel orgs
            'has_religious_authority_military': sourcing_indicators.get('has_religious_authority_military', False),  # Religious + military context
            'is_criticizing_source': sourcing_indicators.get('is_criticizing_source', False),
            'has_direct_action': sourcing_indicators.get('has_direct_action', False),  # Active attribution only
            'has_active_attribution': sourcing_indicators.get('has_active_attribution', False),  # Explicit agency to hegemon
            'has_generic_violence': sourcing_indicators.get('has_generic_violence', False),  # Violence without attribution
            'has_hegemonic_hashtag': sourcing_indicators.get('has_hegemonic_hashtag', False),
            'has_hebrew': sourcing_indicators.get('has_hebrew', False),
            'has_rumor_language': sourcing_indicators.get('has_rumor_language', False),
            'has_us_politicians': sourcing_indicators.get('has_us_politicians', False),
            'has_uk_officials': sourcing_indicators.get('has_uk_officials', False),
            'has_irish_officials': sourcing_indicators.get('has_irish_officials', False),
            'has_denialist_language': sourcing_indicators.get('has_denialist_language', False),
            'has_terror_labeling': sourcing_indicators.get('has_terror_labeling', False),
            'has_genocide_against_palestinians': sourcing_indicators.get('has_genocide_against_palestinians', False),
            'has_genocidal_threat': sourcing_indicators.get('has_genocidal_threat', False),
            'has_counter_hegemonic_page_name': sourcing_indicators.get('has_counter_hegemonic_page_name', False),
            'has_counter_hegemonic_hashtag': sourcing_indicators.get('has_counter_hegemonic_hashtag', False),
            'has_iranian_officials': sourcing_indicators.get('has_iranian_officials', False),
            'has_gaza_sources': sourcing_indicators.get('has_gaza_sources', False),
            'has_al_jazeera': sourcing_indicators.get('has_al_jazeera', False),
            'has_free_palestine': sourcing_indicators.get('has_free_palestine', False),
            'has_dehumanizing_language': sourcing_indicators.get('has_dehumanizing_language', False),
            'has_pro_war_language': sourcing_indicators.get('has_pro_war_language', False),
            'is_palestinian_source': sourcing_indicators.get('is_palestinian_source', False),
            'is_nonprofit_palestinian': sourcing_indicators.get('is_nonprofit_palestinian', False),
            'has_official_stats': linguistic_patterns.get('has_official_stats', False),
            'has_selective_passive_voice': linguistic_patterns.get('has_selective_passive_voice', False),
            'has_command_style': linguistic_patterns.get('has_command_style', False),
            'has_shield': linguistic_patterns.get('has_shield', False),  # The Shield: rationalization patterns
            'has_punitive_language': linguistic_patterns.get('has_punitive_language', False),  # The Sword: punitive language
            'has_appeal_to_leaders': linguistic_patterns.get('has_appeal_to_leaders', False),
            'linguistic_hegemony_score': linguistic_hegemony_score,
            'predicted_category': stance_value if stance_value else None,
            'sourcing_score': sourcing_score,
        }
        
        # Add all zero-shot results
        for i, (label, score) in enumerate(zip(result['labels'], result['scores'])):
            row_data[f'zero_shot_label_{i+1}'] = label
            row_data[f'zero_shot_score_{i+1}'] = score
        
        # Add original row data
        for col in df.columns:
            if col not in row_data:
                row_data[f'csv_{col}'] = row[col] if col in row.index else None
        
        results_data.append(row_data)
    
    results_df = pd.DataFrame(results_data)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("Analysis Summary")
    print(f"{'='*70}")
    print(f"Total texts analyzed: {len(results_df)}")
    print(f"\nZero-Shot Classification Distribution:")
    zero_shot_dist = results_df['zero_shot_label'].value_counts()
    for label, count in zero_shot_dist.items():
        pct = (count / len(results_df)) * 100
        print(f"  {label}: {count} ({pct:.2f}%)")
    
    print(f"\nHegemonic Index (HI) Distribution:")
    print(f"  Thresholds: Hegemony-Maintaining (HI > 0.65), Counter-Hegemonic (HI ≤ 0.4 AND C_res = True),")
    print(f"             Ambivalent/Noise (All others: 0.4 < HI ≤ 0.65 or HI ≤ 0.4 AND C_res = False)")
    hi_dist = results_df['hi_category'].value_counts()
    for category, count in hi_dist.items():
        pct = (count / len(results_df)) * 100
        print(f"  {category}: {count} ({pct:.2f}%)")
    
    print(f"\nHI Score Statistics:")
    print(f"  Mean: {results_df['hi_score'].mean():.4f}")
    print(f"  Median: {results_df['hi_score'].median():.4f}")
    print(f"  Std Dev: {results_df['hi_score'].std():.4f}")
    print(f"  Min: {results_df['hi_score'].min():.4f}")
    print(f"  Max: {results_df['hi_score'].max():.4f}")
    
    print(f"\nHI Component Averages:")
    print(f"  S_n (Stance): {results_df['stance_component'].mean():.4f}")
    print(f"  M_comp (Compositional Moral): {results_df['morals_component'].mean():.4f}")
    print(f"  R_norm (Normalized Reach): {results_df['reach_component'].mean():.4f}")
    print(f"  Z_text (Zero-Shot Hegemony): {results_df['zero_shot_component'].mean():.4f}")
    print(f"  A (Structural Authority): {results_df['authority_component'].mean():.4f}")
    
    print(f"\nInstitutional Sourcing:")
    print(f"  Texts with institutional sources: {results_df['has_institutional_source'].sum()} ({results_df['has_institutional_source'].mean()*100:.2f}%)")
    print(f"  Texts with Israeli sources: {results_df['has_israeli_source'].sum()} ({results_df['has_israeli_source'].mean()*100:.2f}%)")
    print(f"  Texts with US/Western sources: {results_df['has_us_western_source'].sum()} ({results_df['has_us_western_source'].mean()*100:.2f}%)")
    
    print(f"{'='*70}")
    
    return results_df


def find_input_csv(input_csv: str = None, base_path: str = None, additional_files: List[str] = None) -> str:
    """
    Find input CSV file from various possible locations.
    
    Parameters:
    -----------
    input_csv : str, optional
        Explicit path to input CSV file
    base_path : str, optional
        Base directory to search in. If None, uses default.
    additional_files : List[str], optional
        Additional file names to search for (relative to base_path)
    
    Returns:
    --------
    input_csv : str
        Path to found CSV file
    
    Raises:
    -------
    FileNotFoundError
        If no CSV file is found
    """
    if input_csv is not None and os.path.exists(input_csv):
        return input_csv
    
    if base_path is None:
        base_path = '/home/jose/Documents/UNI/Mémoire/Scripts'
    
    # Default files to search for
    default_files = [
        'finaldata_with_sentiment.csv',
        'final_data_with_sentiment.csv',
        'finaldata_with_sentiment_sample_200_random_*.csv',  # Sample files
    ]
    
    if additional_files:
        default_files.extend(additional_files)
    
    possible_files = []
    for file_name in default_files:
        if '*' in file_name:
            pattern = os.path.join(base_path, file_name)
            possible_files.extend(glob.glob(pattern))
        else:
            possible_files.append(os.path.join(base_path, file_name))
    
    lexical_dir = os.path.join(base_path, 'Lexical Analysis')
    if os.path.exists(lexical_dir):
        for file_name in default_files:
            if '*' in file_name:
                pattern = os.path.join(lexical_dir, file_name)
                possible_files.extend(glob.glob(pattern))
            else:
                possible_files.append(os.path.join(lexical_dir, file_name))
    
    possible_files = list(set(possible_files))
    for file_path in possible_files:
        if os.path.exists(file_path):
            return file_path
    
    raise FileNotFoundError(
        f"Could not find CSV file. Searched in:\n"
        f"  Base path: {base_path}\n"
        f"  Files searched: {', '.join(default_files)}\n"
        f"  Tried {len(possible_files)} paths"
    )


def full_run_mode(
    input_csv: str = None,
    output_csv: str = None,
    model_name: str = "facebook/bart-large-mnli",
    device: int = -1,
    batch_size: int = 32,
    additional_files: List[str] = None,
    base_path: str = None
):
    """
    Full run mode: Process all rows and add only hi_category column.
    
    Parameters:
    -----------
    input_csv : str, optional
        Path to input CSV file. If None, tries to find default files
    output_csv : str, optional
        Path to output CSV file. If None, creates a new file with _with_hi_category suffix
    model_name : str
        Hugging Face model name
    device : int
        Device to use (-1 for CPU, 0+ for GPU)
    batch_size : int
        Batch size for processing
    additional_files : List[str], optional
        Additional file names to search for if input_csv is None
    base_path : str, optional
        Base directory to search for files. If None, uses default.
    """
    print("="*70)
    print("Full Run Mode - Adding hi_category to all rows")
    print("="*70)
    
    try:
        input_csv = find_input_csv(input_csv, base_path, additional_files)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\nLoading data from: {input_csv}")
    try:
        df = pd.read_csv(input_csv, encoding='utf-8', low_memory=False)
        if 'constructed_text' not in df.columns:
            df = pd.read_csv(input_csv, sep=';', encoding='utf-8', low_memory=False)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    print(f"  Loaded {len(df)} rows")
    
    if 'constructed_text' not in df.columns:
        print("Error: 'constructed_text' column not found!")
        sys.exit(1)
    
    stance_col = 'predicted_category' if 'predicted_category' in df.columns else None
    authority_col = 'csv_Authority' if 'csv_Authority' in df.columns else ('Authority' if 'Authority' in df.columns else None)
    loyalty_col = 'csv_Loyalty' if 'csv_Loyalty' in df.columns else ('Loyalty' if 'Loyalty' in df.columns else None)
    
    reach_cols = []
    for col in ['Followers at Posting', 'Likes at Posting', 'Total Interactions']:
        if f'csv_{col}' in df.columns:
            reach_cols.append(f'csv_{col}')
        elif col in df.columns:
            reach_cols.append(col)
    
    print(f"\nDetected columns:")
    print(f"  Stance: {stance_col}")
    print(f"  Authority: {authority_col}")
    print(f"  Loyalty: {loyalty_col}")
    print(f"  Reach: {reach_cols}")
    
    print(f"\nProcessing all {len(df)} rows...")
    print("  This may take a while depending on dataset size...")
    
    results_df = analyze_hegemonic_positions_doxa(
        df,
        text_column='constructed_text',
        stance_column=stance_col or 'predicted_category',
        authority_column=authority_col or 'csv_Authority',
        loyalty_column=loyalty_col or 'csv_Loyalty',
        reach_columns=reach_cols if reach_cols else ['csv_Followers at Posting', 'csv_Likes at Posting', 'csv_Total Interactions'],
        model_name=model_name,
        batch_size=batch_size,
        max_texts=None,  # Process all rows
        device=device
    )
    
    print(f"\nExtracting hi_category column...")
    
    hi_category_map = dict(zip(results_df['text_index'], results_df['hi_category']))
    df['hi_category'] = df.index.map(lambda idx: hi_category_map.get(idx, 'Unknown'))
    
    mapped_count = df['hi_category'].value_counts().get('Unknown', 0)
    if mapped_count > 0:
        print(f"  Warning: {mapped_count} rows could not be mapped (marked as 'Unknown')")
    else:
        print(f"  ✓ Successfully mapped hi_category for all {len(df)} rows")
    
    if output_csv is None:
        base_name = os.path.splitext(input_csv)[0]
        output_csv = f"{base_name}_with_hi_category.csv"
    
    print(f"\nSaving results to: {output_csv}")
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Total rows processed: {len(df)}")
    print(f"\nHI Category Distribution:")
    hi_dist = df['hi_category'].value_counts()
    for category, count in hi_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {category}: {count} ({pct:.2f}%)")
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*70}")


def main(
    input_csv: str = None,
    output_dir: str = None,
    model_name: str = None,
    device: int = None,
    batch_size: int = None,
    max_texts: int = None,
    additional_files: List[str] = None,
    base_path: str = None
):
    """
    Main function.
    
    Parameters:
    -----------
    input_csv : str, optional
        Path to input CSV file. If None, tries to find default files
    output_dir : str, optional
        Output directory for results. If None, uses default.
    model_name : str, optional
        Hugging Face model name. If None, uses default.
    device : int, optional
        Device to use (-1 for CPU, 0+ for GPU). If None, uses default.
    batch_size : int, optional
        Batch size for processing. If None, uses default.
    max_texts : int, optional
        Maximum number of texts to process. If None, processes all.
    additional_files : List[str], optional
        Additional file names to search for if input_csv is None
    base_path : str, optional
        Base directory to search for files. If None, uses default.
    """
    # Configuration
    if base_path is None:
        base_path = '/home/jose/Documents/UNI/Mémoire/Scripts'
    
    try:
        input_csv = find_input_csv(input_csv, base_path, additional_files)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if output_dir is None:
        output_dir = '/home/jose/Documents/UNI/Mémoire/Scripts/hegemonic_positions_results'
    os.makedirs(output_dir, exist_ok=True)
    
    if model_name is None:
        model_name = "facebook/bart-large-mnli"
    if device is None:
        device = -1
    
    if batch_size is None:
        batch_size = 32
    if max_texts is None:
        max_texts = None
    
    print("="*70)
    print("Zero-Shot Hegemonic Position Analysis - Doxa Framework")
    print("="*70)
    print("\nFormula: HI = ω₁(S_n) + ω₂(M_comp) + ω₃(R_norm) + ω₄(Z_text) + ω₅(A)")
    print("  Weights: ω₁ = 0.15 (Stance), ω₂ = 0.25 (Moral Composition),")
    print("           ω₃ = 0.2 (Structural Access/Reach), ω₄ = 0.2 (Linguistic Analysis),")
    print("           ω₅ = 0.2 (Structural Authority)")
    print("\n  Components:")
    print("    S_n: Stance score (1.0 for Pro-Israeli, 0.5 for Neutral, 0.0 for Pro-Palestinian)")
    print("    M_comp: (Authority + Loyalty) / sum(all moral foundations)")
    print("    R_norm: log(Reach_i + 1) / max(log(Reach))")
    print("    Z_text: Zero-Shot probability for 'hegemonic' label")
    print("    A: Structural Authority Score (based on Page Category and Page Admin Top Country)")
    print("\n  Classification:")
    print("    Hegemony-Maintaining (HI > 0.65): Naturalizes dominant power relations through")
    print("                                       high access, authority morals, structural authority")
    print("    Counter-Hegemonic (HI ≤ 0.4 AND C_res = True): Actively challenges power through")
    print("                                                    moral critique, critical terminology, or alignment")
    print("    Ambivalent / Noise (All others): Transitional (0.4 < HI ≤ 0.65) or")
    print("                                      Passive/Irrelevant (HI ≤ 0.4 AND C_res = False)")
    print("="*70)
    
    print(f"\nLoading data from: {input_csv}")
    try:
        df = pd.read_csv(input_csv, encoding='utf-8', low_memory=False)
        if 'constructed_text' not in df.columns:
            df = pd.read_csv(input_csv, sep=';', encoding='utf-8', low_memory=False)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    print(f"  Loaded {len(df)} rows")
    
    if 'constructed_text' not in df.columns:
        print("Error: 'constructed_text' column not found!")
        sys.exit(1)
    
    stance_col = 'predicted_category' if 'predicted_category' in df.columns else None
    authority_col = 'csv_Authority' if 'csv_Authority' in df.columns else ('Authority' if 'Authority' in df.columns else None)
    loyalty_col = 'csv_Loyalty' if 'csv_Loyalty' in df.columns else ('Loyalty' if 'Loyalty' in df.columns else None)
    
    reach_cols = []
    for col in ['Followers at Posting', 'Likes at Posting', 'Total Interactions']:
        if f'csv_{col}' in df.columns:
            reach_cols.append(f'csv_{col}')
        elif col in df.columns:
            reach_cols.append(col)
    
    print(f"\nDetected columns:")
    print(f"  Stance: {stance_col}")
    print(f"  Authority: {authority_col}")
    print(f"  Loyalty: {loyalty_col}")
    print(f"  Reach: {reach_cols}")
    
    # Perform analysis
    results_df = analyze_hegemonic_positions_doxa(
        df,
        text_column='constructed_text',
        stance_column=stance_col or 'predicted_category',
        authority_column=authority_col or 'csv_Authority',
        loyalty_column=loyalty_col or 'csv_Loyalty',
        reach_columns=reach_cols if reach_cols else ['csv_Followers at Posting', 'csv_Likes at Posting', 'csv_Total Interactions'],
        model_name=model_name,
        batch_size=batch_size,
        max_texts=max_texts,
        device=device
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'hegemonic_positions_doxa_{timestamp}.csv')
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    
    summary_file = os.path.join(output_dir, f'hegemonic_positions_doxa_summary_{timestamp}.json')
    summary = {
        'total_texts': len(results_df),
        'hi_score_mean': float(results_df['hi_score'].mean()),
        'hi_score_std': float(results_df['hi_score'].std()),
        'hi_category_distribution': results_df['hi_category'].value_counts().to_dict(),
        'zero_shot_distribution': results_df['zero_shot_label'].value_counts().to_dict(),
        'doxa_component_averages': {
            'stance': float(results_df['stance_component'].mean()),
            'morals': float(results_df['morals_component'].mean()),
            'reach': float(results_df['reach_component'].mean()),
            'zero_shot': float(results_df['zero_shot_component'].mean()),
            'authority': float(results_df['authority_component'].mean())
        },
        'model_used': model_name,
        'hypothesis_template': HYPOTHESIS_TEMPLATE,
        'candidate_labels': CANDIDATE_LABELS
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*70}")
    print("Analysis complete!")


def test_mode(
    model_name: str = "facebook/bart-large-mnli",
    device: int = -1,
    num_random_texts: int = 10,
    input_csv: str = None,
    additional_files: List[str] = None,
    base_path: str = None
):
    """
    Test mode for Doxa-based hegemonic position analysis.
    Tests on random texts from the dataset.
    
    Parameters:
    -----------
    model_name : str
        Hugging Face model name
    device : int
        Device to use (-1 for CPU, 0+ for GPU)
    num_random_texts : int
        Number of random texts to test
    input_csv : str, optional
        Path to input CSV file. If None, tries to find default files
    additional_files : List[str], optional
        Additional file names to search for if input_csv is None
    base_path : str, optional
        Base directory to search for files. If None, uses default.
    """
    print("="*70)
    print("TEST MODE - Hegemonic Position Analysis (Doxa Framework)")
    print("="*70)
    print("\nDOXA = Stance + Morals + Reach + Sourcing")
    print("HI = (Neutral Stance) + (Authority/Loyalty) + (High Reach) + (Zero-Shot Score)")
    print("="*70)
    
    print(f"\nLoading {num_random_texts} random texts...")
    
    try:
        input_csv = find_input_csv(input_csv, base_path, additional_files)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    try:
        df = pd.read_csv(input_csv, encoding='utf-8', low_memory=False)
        if 'constructed_text' not in df.columns:
            df = pd.read_csv(input_csv, sep=';', encoding='utf-8', low_memory=False)
        
        if 'constructed_text' not in df.columns:
            print("Error: 'constructed_text' column not found!")
            return
        
        df = df[df['constructed_text'].notna()]
        df = df[df['constructed_text'].astype(str).str.strip() != '']
        
        if len(df) == 0:
            print("Error: No valid texts found!")
            return
        
        num_to_sample = min(num_random_texts, len(df))
        sampled_df = df.sample(n=num_to_sample, random_state=None).reset_index(drop=True)
        
        print(f"  ✓ Loaded {len(sampled_df)} random texts")
        
        results_df = analyze_hegemonic_positions_doxa(
            sampled_df,
            text_column='constructed_text',
            stance_column='predicted_category',
            authority_column='csv_Authority',
            loyalty_column='csv_Loyalty',
            reach_columns=['csv_Followers at Posting', 'csv_Likes at Posting', 'csv_Total Interactions'],
            model_name=model_name,
            batch_size=32,
            max_texts=None,
            device=device
        )
        
        output_dir = '/home/jose/Documents/UNI/Mémoire/Scripts/hegemonic_positions_results'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f'hegemonic_positions_doxa_test_{timestamp}.csv')
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n{'='*70}")
        print(f"Test results saved to: {output_file}")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Zero-Shot Hegemonic Position Analysis with Doxa Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (tests on random sample instead of full dataset)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='facebook/bart-large-mnli',
        help='Hugging Face model name (default: facebook/bart-large-mnli)'
    )
    
    parser.add_argument(
        '--device',
        type=int,
        default=-1,
        help='Device to use (-1 for CPU, 0+ for GPU, default: -1)'
    )
    
    parser.add_argument(
        '--num-texts',
        type=int,
        default=10,
        help='Number of random texts for test mode (default: 10)'
    )
    
    parser.add_argument(
        '--full-run',
        action='store_true',
        help='Run in full mode: process all rows and add only hi_category column'
    )
    
    parser.add_argument(
        '--input-csv',
        type=str,
        default=None,
        help='Input CSV file path (for full-run mode)'
    )
    
    parser.add_argument(
        '--output-csv',
        type=str,
        default=None,
        help='Output CSV file path (for full-run mode)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (for main mode)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for processing (default: 32 for CPU, 64 for GPU)'
    )
    
    parser.add_argument(
        '--max-texts',
        type=int,
        default=None,
        help='Maximum number of texts to process (default: all texts)'
    )
    
    parser.add_argument(
        '--additional-files',
        type=str,
        nargs='+',
        default=None,
        help='Additional file names to search for if input CSV is not specified'
    )
    
    parser.add_argument(
        '--base-path',
        type=str,
        default=None,
        help='Base directory to search for files (default: /home/jose/Documents/UNI/Mémoire/Scripts)'
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_mode(
            model_name=args.model,
            device=args.device,
            num_random_texts=args.num_texts,
            input_csv=args.input_csv,
            additional_files=args.additional_files,
            base_path=args.base_path
        )
    elif args.full_run:
        full_run_mode(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            additional_files=args.additional_files,
            base_path=args.base_path
        )
    else:
        main(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            max_texts=args.max_texts,
            additional_files=args.additional_files,
            base_path=args.base_path
        )

