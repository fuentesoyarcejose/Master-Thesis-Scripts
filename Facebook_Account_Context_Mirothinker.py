"""
Account Context Finder using MiroThinker Model

This script analyzes Facebook account information using the MiroThinker-v1.5-30B model
from Hugging Face to extract organizational context variables.

Dependencies:
- transformers: For model loading
- torch: Required by transformers
- pandas: For CSV processing

Usage:
    # Direct model loading
    python account_context_openAI.py data.csv
    
    # With quantization (reduces memory)
    python account_context_openAI.py data.csv --load-in-4bit
"""

import pandas as pd
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

prompt_template = """You are “AccountContextFinder”.
Task: Find context and information from social media pages (specifically Facebook) with the information provided

Think step-by-step **silently** (don't reveal your reasoning).  

Format your response as:
1. Legal Nature: [value]
2. Profit Status: [value]
3. Ownership: [value]
4. Sector: [value]
5. Type: [value]
6. Subtype: [value]
7. Public Authority: [value]
8. State affiliation: [value]
9. Core Function: [value]


### Definitions of variables

**Legal Nature*
Identify the legal status of the entity according to how it is constituted under law, look for whether the entity is legally public, private, or hybrid.

**Profit Status**
Determine whether the entity operates for profit or not, based on its stated mission and legal status. Look for: Distribution of profits to owners/shareholders, Charitable or commercial orientation.

Allowed values:
For-profit
Non-profit
Mixed / dual (non-profit parent with commercial subsidiaries)

**Ownership**
Identify who ultimately owns or controls the entity. Look for state ownership, private shareholders, parent organizations, membership or foundation ownership

Allowed values:
State-owned
Privately owned
Publicly owned (no shareholders)
Foundation / association owned
Subsidiary of another organization

**Sector**
Classify the entity according to the broad institutional sector it belongs to. Look for: Relationship to state, market, or civil society.

Allowed values:
Public sector
Private market sector
Third sector / civil society
Public service media sector
Hybrid sector

Definition of values:

Public sector: Organizations that are part of the state apparatus and exercise public authority.
Private market sector: Organizations primarily operating according to market logic and profit maximization.
Third sector / civil society: Non-state, non-market organizations oriented toward social, religious, or civic goals.
Public service media sector: Organizations with a public mandate, created or chartered by the state, but institutionally independent and not part of government.
Hybrid sector: Organizations combining multiple sector logics, with no single dominant one.

Logic reasoning:

Rule 1: If the entity has public authority, it belongs to the Public Sector.
Rule 2: If profit generation is the primary organizing logic, it belongs to the Private Market Sector.
Rule 3: If it is non-profit, non-state, and mission-driven, it belongs to the Third Sector / Civil Society.
Rule 4: If it has a public mandate without government control, classify it as Public Service Media.
Rule 5: If two logics are equally central, use Hybrid Sector.
Rule 6: If you cant defined with the previously donated rules, categorize as NA

**Type**

Identify the general organizational category of the entity. Whether it is an institution, company, organization, or individual

Allowed Values:
Public institution
Private company
Non-governmental organization (NGO)
Media organization
Individual / public figure
Religious organization

**Subtype**

Specify the functional or thematic subtype within the broader type. Meaning the domain of activity and/or specialization you are allowed fit the value most adecuate for this variable

**Public Authority**

Assess whether the entity has formal public authority or statutory power granted by the state. Look for legal authority, regulatory or governing powers. It is about authority, not influence. Authority = the right to decide, regulate, or enforce. Influence = the capacity to persuade or shape opinion

Allowed values:
Yes
No
Limited / delegated

Logic reasoning:
Rule 1: If the entity can legally regulate, enforce, or sanction → Yes
Rule 2: If it only informs, advises, reports, or advocates → No
Rule 3: If authority exists only in a narrowly defined delegated role → Limited / delegated
Rule 4: Do NOT infer authority from funding, prestige, or reach.

**State Affiliation**ù
Determine the degree of institutional affiliation with the state. Look for: creation by law or charter, public funding, government oversight or appointments

Allowed values:
None
Indirect (funding, charter, regulation)
Direct (state-controlled or state-run)

**Core Function**

Identify the primary function or role the entity performs in society. Look for: Main activity, not secondary ones, 
Stated mission or observable output

Allowed values (examples):
News production
Broadcasting
Humanitarian aid
Religious services
Political advocacy
Commercial content distribution
Education / research

In this option look for information in the sources, they have to match with the facebook account. Do not infer or guess missing information according to the name of the account, for example, if the name of the account is "The Guardian" and the core function is "News production", do not add "Media organization" or "News organization" to the value not "security" or "defense". 

###Search & Verification Instructions (CRITICAL)

1. Primary source priority

First, search for information directly associated with the Facebook ID.
Use the Facebook Page’s About section, Page Transparency, linked websites, and publicly available metadata.

2. Name and username validation
Use the Name and Username to search for external information only if it clearly refers to the same entity as the Facebook account.
Do not assume equivalence based on name similarity alone.

3. External link validation (VERY IMPORTANT)
The Link must only be used if it is explicitly linked to or claimed by the Facebook account (e.g. listed in the About section, bio, or posts).
Do NOT infer identity from a generic domain.
	Example: a link to youtube.com does not imply the account is YouTube.
	Only use the link if the Facebook account clearly represents or officially operates that site or channel.

4. Context consistency check
Before using any external source, verify that:
	The organization name, branding, and function match the Facebook account.
	There is clear evidence of ownership, official representation, or direct affiliation.
If context cannot be confirmed, do not use the source.

5. Source quality requirement
Prioritize:
	Official websites
	Government or registry records
	Recognized institutional pages (e.g. Wikipedia, corporate filings, official “About” pages)
Avoid blogs, opinion sites, or unrelated aggregators unless explicitly cited by the organization itself.


###Output Rules
Base each classification on verifiable evidence.
Do not infer or guess missing information.
If information is insufficient or ambiguous, explicitly state:
	“Insufficient verified information to determine.”"""

# Global model and tokenizer variables (loaded once)
_model = None
_tokenizer = None

def load_mirothinker_model(model_name: str = "miromind-ai/MiroThinker-v1.5-30B", 
                          device_map: str = "auto",
                          load_in_8bit: bool = False,
                          load_in_4bit: bool = False):
    """Load MiroThinker model and tokenizer. Call this once at the start."""
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    print(f"Loading MiroThinker model: {model_name}...")
    print("This may take a few minutes on first run...")
    
    try:
        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Prepare model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "torch_dtype": torch.bfloat16,
        }
        
        # Add quantization if requested
        if load_in_8bit or load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = quantization_config
        
        # Load model
        _model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        _model.eval()
        
        print("Model loaded successfully!")
        return _model, _tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nNote: For large models like MiroThinker, consider:")
        print("1. Using quantization (load_in_4bit=True or load_in_8bit=True)")
        print("2. Ensuring you have sufficient GPU memory (30B model needs ~60GB+ VRAM without quantization)")
        raise

def ask_mirothinker(facebook_id, page_name, username, external_link, 
                   model_name: str = "miromind-ai/MiroThinker-v1.5-30B",
                   temperature: float = 0.2,
                   max_tokens: int = 1000):
    """
    Query MiroThinker model for account context analysis.
    
    Args:
        facebook_id: Facebook ID
        page_name: Page name
        username: Username
        external_link: External link
        model_name: Hugging Face model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    """
    user_data = f"""\n\n###Data\n\nFor this You are given the following information about a Facebook account:\n\nFacebook ID: {facebook_id}\n\nName: {page_name}\n\nUsername: {username}\n\nLink: {external_link}\n"""
    prompt = prompt_template + user_data
    
    # Add system instruction
    full_prompt = f"""You are AccountContextFinder. 

IMPORTANT: Return ALL nine variables in the exact format specified:
1. Legal Nature: [value]
2. Profit Status: [value]
3. Ownership: [value]
4. Sector: [value]
5. Type: [value]
6. Subtype: [value]
7. Public Authority: [value]
8. State affiliation: [value]
9. Core Function: [value]

If information is insufficient for any variable, state 'Insufficient verified information to determine.' for that field.
Do not skip any variables. Return all nine variables.

{prompt}"""
    
    # Use direct model inference
    global _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        load_mirothinker_model(model_name)
    
    try:
        # Format prompt for the model
        # MiroThinker uses a specific chat template, so we'll use it if available
        if hasattr(_tokenizer, 'apply_chat_template') and _tokenizer.chat_template:
            messages = [{"role": "user", "content": full_prompt}]
            formatted_prompt = _tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            formatted_prompt = full_prompt
        
        # Tokenize
        inputs = _tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=256000)
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=_tokenizer.eos_token_id,
                eos_token_id=_tokenizer.eos_token_id,
            )
        
        # Decode response
        generated_text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the prompt)
        if formatted_prompt in generated_text:
            response = generated_text.split(formatted_prompt, 1)[1].strip()
        else:
            response = generated_text.strip()
        
        return response
        
    except Exception as e:
        return f"Error: {e}"

def parse_variables(response):
    """
    Parse variables from model response with multiple parsing strategies.
    Returns dict with all variables and the raw response.
    """
    items = ['Legal Nature', 'Profit Status', 'Ownership', 'Sector', 'Type', 'Subtype',
             'Public Authority', 'State affiliation', 'Core Function']
    result = {k: '' for k in items}
    result['Raw Response'] = response if response else ''
    
    if response is None or not isinstance(response, str) or not response.strip():
        return result
    
    # Strategy 1: Look for numbered list format (1. Variable Name: value)
    nums = ('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    
    # Check if response uses numbered format
    numbered_format = any(l.startswith(n) for l in lines for n in nums)
    
    if numbered_format:
        for i, key in enumerate(items, 1):
            for line in lines:
                # Match patterns like "1. Legal Nature: value" or "1. Legal Nature - value"
                if line.startswith(f"{i}."):
                    # Try colon separator first
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) >= 2:
                            result[key] = parts[1].strip()
                            break
                    # Try dash separator
                    elif '-' in line:
                        parts = line.split('-', 1)
                        if len(parts) >= 2:
                            result[key] = parts[1].strip()
                            break
                    # Just remove the number prefix
                    else:
                        cleaned = line.split('.', 1)[1].strip()
                        result[key] = cleaned.strip(": \t-")
                        break
    
    # Strategy 2: Look for "Variable Name: value" format (case-insensitive)
    if not numbered_format or any(v == '' for v in result.values() if v != result['Raw Response']):
        for line in lines:
            line_lower = line.lower()
            for key in items:
                key_lower = key.lower()
                # Check if line contains the key name
                if key_lower in line_lower:
                    # Try colon separator
                    if ':' in line:
                        # Make sure the key appears before the colon
                        colon_idx = line.find(':')
                        key_idx = line_lower.find(key_lower)
                        if key_idx < colon_idx:
                            parts = line.split(':', 1)
                            if len(parts) >= 2:
                                value = parts[1].strip()
                                # Only update if we haven't found a value yet or this is more specific
                                if result[key] == '' or len(value) > len(result[key]):
                                    result[key] = value
                    # Try dash separator
                    elif '-' in line:
                        dash_idx = line.find('-')
                        key_idx = line_lower.find(key_lower)
                        if key_idx < dash_idx:
                            parts = line.split('-', 1)
                            if len(parts) >= 2:
                                value = parts[1].strip()
                                if result[key] == '' or len(value) > len(result[key]):
                                    result[key] = value
    
    # Strategy 3: Look for patterns like "Legal Nature = value" or "Legal Nature: value" on same line
    for key in items:
        if result[key] == '':
            key_lower = key.lower()
            for line in lines:
                line_lower = line.lower()
                if key_lower in line_lower:
                    # Try "=" separator
                    if '=' in line:
                        parts = line.split('=', 1)
                        if len(parts) >= 2 and key_lower in parts[0].lower():
                            result[key] = parts[1].strip()
                            break
    
    return result

def main(input_file, model_name="miromind-ai/MiroThinker-v1.5-30B", 
         load_in_4bit=False, load_in_8bit=False, temperature=0.2, max_tokens=1000):
    """
    Main function to process CSV file with MiroThinker.
    
    Args:
        input_file: Path to input CSV file
        model_name: Hugging Face model name
        load_in_4bit: Use 4-bit quantization (reduces memory usage)
        load_in_8bit: Use 8-bit quantization (reduces memory usage)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    """
    df = pd.read_csv(input_file)
    req_cols = ["facebook_id","page_name","username","external_link"]
    for col in req_cols:
        if col not in df.columns:
            raise Exception(f"Missing column: {col}")
    res_vars = ['Legal Nature', 'Profit Status', 'Ownership', 'Sector', 'Type', 'Subtype',
             'Public Authority', 'State affiliation', 'Core Function']
    for v in res_vars:
        if v not in df.columns:
            df[v] = ""
    
    # Add raw response column for debugging
    if 'Raw Response' not in df.columns:
        df['Raw Response'] = ""
    
    # Load model
    print("Loading MiroThinker model (this may take a few minutes)...")
    try:
        load_mirothinker_model(model_name, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nTips:")
        print("- Use --load-in-4bit to reduce memory usage")
        print("- Ensure you have sufficient GPU memory (30B model needs ~60GB+ VRAM without quantization)")
        return
    
    print(f"Processing {len(df)} rows...")
    for idx, row in df.iterrows():
        facebook_id = row.get("facebook_id","")
        page_name = row.get("page_name","")
        username = row.get("username","")
        external_link = row.get("external_link","")
        
        print(f"Processing row {idx + 1}/{len(df)}: {page_name}")
        
        chat_resp = ask_mirothinker(
            facebook_id, page_name, username, external_link,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        parsed = parse_variables(chat_resp)
        for v in res_vars:
            df.at[idx, v] = parsed.get(v, "")
        
        # Save raw response for debugging
        df.at[idx, 'Raw Response'] = parsed.get('Raw Response', "")
        
        # Print parsed results for verification
        print(f"  Parsed {sum(1 for v in res_vars if parsed.get(v, ''))}/{len(res_vars)} variables")
        
        # Save progress periodically
        if (idx + 1) % 10 == 0:
            df.to_csv(input_file, index=False)
            print(f"Progress saved at row {idx + 1}")
    
    df.to_csv(input_file, index=False)
    print(f"Processing complete! Results saved to {input_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze Facebook account context using MiroThinker model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct model loading (requires sufficient GPU memory)
  python account_context_openAI.py data.csv
  
  # With 4-bit quantization (reduces memory usage)
  python account_context_openAI.py data.csv --load-in-4bit
        """
    )
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--model-name", type=str, default="miromind-ai/MiroThinker-v1.5-30B",
                       help="Hugging Face model name (default: miromind-ai/MiroThinker-v1.5-30B)")
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="Use 4-bit quantization to reduce memory usage")
    parser.add_argument("--load-in-8bit", action="store_true",
                       help="Use 8-bit quantization to reduce memory usage")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Sampling temperature (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=1000,
                       help="Maximum tokens to generate (default: 1000)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)
    
    if args.load_in_4bit and args.load_in_8bit:
        print("Error: Cannot use both 4-bit and 8-bit quantization")
        sys.exit(1)
    
    main(
        args.input_file,
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )


