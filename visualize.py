import streamlit as st
import json
import re
import html
import os
from typing import List, Dict, Any

@st.cache_data
def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads JSON data from the specified file path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def calculate_hits_at_k(file_path: str, ks: tuple = (1, 5, 10, 100)):
    """Calculates Hits@k metrics for the dataset efficiently."""
    dataset = load_data(file_path)
    hits = {k: 0 for k in ks}
    total = len(dataset)
    if total == 0:
        return hits
    
    max_k = max(ks)
    for item in dataset:
        positive_ids = {p.get('passage_id') or p.get('text', '') for p in item.get('positive_ctxs', [])}
        
        # Find the rank of the first hit in a single pass
        for i, ctx in enumerate(item.get('ctxs', [])[:max_k]):
            if (ctx.get('passage_id') or ctx.get('text', '')) in positive_ids:
                rank = i + 1
                for k in ks:
                    if rank <= k:
                        hits[k] += 1
                break
    
    return {k: (v / total) * 100 for k, v in hits.items()}

def highlight_text(text: str, keys: List[List[Any]], color_mapping: Dict[str, Dict[str, str]]) -> str:
    """Highlights key matches in the text using HTML <mark> tags with specific colors and score on hover."""
    if not keys:
        return html.escape(text)
    
    # Create a local mapping for scores in this context
    # Use lowercase for case-insensitive lookup
    score_mapping = {}
    for k in keys:
        if isinstance(k, list) and len(k) >= 3:
            key_text = str(k[0]).strip().lower()
            score = k[2]
            # Keep the highest score if the same key appears multiple times in one context
            if key_text not in score_mapping or score > score_mapping[key_text]:
                score_mapping[key_text] = score

    # Sort keys by length descending to handle overlapping matches
    sorted_keys = sorted([str(k[0]).strip() for k in keys], key=len, reverse=True)
    
    # Remove duplicates
    unique_keys = []
    for k in sorted_keys:
        if k not in unique_keys and k:
            unique_keys.append(k)
            
    if not unique_keys:
        return html.escape(text)

    # Escape HTML in the original text first
    text = html.escape(text)
    
    # Create a regex pattern to match any of the keys (case-insensitive)
    pattern = "|".join([re.escape(html.escape(k)) for k in unique_keys])
    
    def replace_func(match):
        val_original = match.group(0)
        val_lower = val_original.strip().lower()
        style = color_mapping.get(val_lower, {"bg": "#ffff00", "fg": "black"})
        score = score_mapping.get(val_lower, "N/A")
        
        # Format score to 2 decimal places if it's a number
        score_str = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
        
        return f'<mark title="Score: {score_str}" style="background-color: {style["bg"]}; color: {style["fg"]}; padding: 2px; border-radius: 2px; cursor: help;">{val_original}</mark>'
    
    # Use re.sub with a case-insensitive flag
    highlighted = re.sub(f"({pattern})", replace_func, text, flags=re.IGNORECASE)
    return highlighted

def main():
    st.set_page_config(page_title="IR Data Visualization Tool", layout="wide")
    st.title("🔍 Information Retrieval Data Visualizer")

    # Sidebar for file selection and configuration
    st.sidebar.header("Data Selection")
    
    # Get datasets (top-level directories, excluding venv and dotfiles)
    datasets = sorted([d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.') and d != 'venv'])
    
    if not datasets:
        st.sidebar.error("No dataset directories found.")
        return
        
    # Default selection
    default_dataset = "LIMIT"
    ds_idx = datasets.index(default_dataset) if default_dataset in datasets else 0
    selected_dataset = st.sidebar.selectbox("Dataset", datasets, index=ds_idx)
    
    # Get decoding algorithms for the selected dataset
    ds_path = os.path.join('.', selected_dataset)
    algorithms = sorted([d for d in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, d))])
    
    if not algorithms:
        st.sidebar.warning(f"No algorithm directories found in {selected_dataset}.")
        return
        
    default_algo = "BEAM"
    algo_idx = algorithms.index(default_algo) if default_algo in algorithms else 0
    selected_algo = st.sidebar.selectbox("Decoding Algorithm", algorithms, index=algo_idx)
    
    # Get JSON files for the selected algorithm
    algo_path = os.path.join(ds_path, selected_algo)
    json_files = sorted([f for f in os.listdir(algo_path) if f.endswith('.json')])
    
    if not json_files:
        st.sidebar.warning(f"No JSON files found in {selected_algo}.")
        return
        
    default_file = "output_seal_limit_beam_intersect.json"
    file_idx = json_files.index(default_file) if default_file in json_files else 0
    selected_file = st.sidebar.selectbox("Output File", json_files, index=file_idx)
    
    file_path = os.path.join(algo_path, selected_file)
    
    try:
        data = load_data(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return

    # Calculate Hits@k for the dataset (cached)
    hits_metrics = calculate_hits_at_k(file_path, ks=(1, 5, 10, 100))
    
    # Display Metrics at the top
    st.subheader(f"📊 Global Metrics: {selected_file}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Queries", len(data))
    m2.metric("Hits@1", f"{hits_metrics[1]:.1f}%")
    m3.metric("Hits@5", f"{hits_metrics[5]:.1f}%")
    m4.metric("Hits@10", f"{hits_metrics[10]:.1f}%")
    m5.metric("Hits@100", f"{hits_metrics[100]:.1f}%")
    
    st.divider()

    st.sidebar.write(f"Total Queries: {len(data)}")
    
    # Search and Filter
    query_search = st.sidebar.text_input("Search Question")
    filtered_data = [d for d in data if query_search.lower() in d['question'].lower()]
    
    if not filtered_data:
        st.warning("No queries match the search criteria.")
        return

    def has_top_positive(item):
        p_ids = set()
        for p in item.get('positive_ctxs', []):
            p_ids.add(p.get('passage_id') or p.get('text', ''))
        
        for ctx in item.get('ctxs', [])[:2]:
            if (ctx.get('passage_id') or ctx.get('text', '')) in p_ids:
                return True
        return False

    # Select query
    selected_idx = st.sidebar.selectbox(
        "Select Question", 
        range(len(filtered_data)), 
        format_func=lambda i: f"{'✅ ' if has_top_positive(filtered_data[i]) else ''}{i+1}: {filtered_data[i]['question'][:50]}..."
    )
    
    item = filtered_data[selected_idx]
    
    # Pre-calculate key categories and color mapping
    positive_ids = set()
    for p in item.get('positive_ctxs', []):
        positive_ids.add(p.get('passage_id') or p.get('text', ''))

    pos_keys = {} # key_text -> max_score
    neg_keys = {} # key_text -> max_score
    
    for ctx in item.get('ctxs', []):
        is_pos = (ctx.get('passage_id') or ctx.get('text', '')) in positive_ids
                
        keys_data = ctx.get('keys', [])
        if isinstance(keys_data, str):
            try: keys_data = json.loads(keys_data)
            except: keys_data = []
        
        for k in keys_data:
            key_text = ""
            score = 0.0
            if isinstance(k, list) and k:
                key_text = str(k[0]).strip()
                if len(k) >= 3: score = k[2]
            elif isinstance(k, str):
                key_text = k.strip()
            
            if key_text:
                target_dict = pos_keys if is_pos else neg_keys
                # Update max score for this key text
                if key_text not in target_dict or score > target_dict[key_text]:
                    target_dict[key_text] = score
    
    color_mapping = {} # key_lower -> {bg, fg}
    
    all_keys_with_scores = {} # key_text -> (max_score, category)
    
    # Identify unique keys across both positive and negative
    unique_key_texts = set(pos_keys.keys()) | set(neg_keys.keys())
    
    for k in unique_key_texts:
        is_in_pos = k in pos_keys
        is_in_neg = k in neg_keys
        max_score = max(pos_keys.get(k, 0.0), neg_keys.get(k, 0.0))
        
        if is_in_pos and is_in_neg:
            category = "both"
            color_mapping[k.lower()] = {"bg": "#ffc107", "fg": "black"}
        elif is_in_pos:
            category = "only_pos"
            color_mapping[k.lower()] = {"bg": "#28a745", "fg": "white"}
        else:
            category = "only_neg"
            color_mapping[k.lower()] = {"bg": "#dc3545", "fg": "white"}
            
        all_keys_with_scores[k] = (max_score, category)

    # Sort all keys by max score descending for the summary
    sorted_summary_keys = sorted(all_keys_with_scores.items(), key=lambda x: x[1][0], reverse=True)
    
    # Display Question and Answers
    st.header(f"Question: {item['question']}")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.subheader("Expected Answers")
        for ans in item.get('answers', []):
            st.markdown(f"- {ans}")
            
    with col2:
        st.subheader("Stats")
        st.write(f"Total retrieved (ctxs): {len(item.get('ctxs', []))}")
        st.write(f"Total positives: {len(item.get('positive_ctxs', []))}")

    with col3:
        st.subheader("All Keys Matched")
        if not sorted_summary_keys:
            st.write("None")
        else:
            badges = []
            for k, (score, cat) in sorted_summary_keys:
                style = color_mapping[k.lower()]
                score_str = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
                badges.append(f'<span title="Max Score: {score_str}" style="background-color: {style["bg"]}; color: {style["fg"]}; padding: 2px 6px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 4px; cursor: help;">{k}</span>')
            
            st.markdown(f'<div>{" ".join(badges)}</div>', unsafe_allow_html=True)

    st.divider()

    # Display Retrieved Documents
    st.subheader("Retrieved Documents (ctxs)")
    
    for i, ctx in enumerate(item.get('ctxs', [])):
        is_positive = (ctx.get('passage_id') or ctx.get('text', '')) in positive_ids
            
        keys_data = ctx.get('keys', [])
        if isinstance(keys_data, str):
            try: keys_data = json.loads(keys_data)
            except: keys_data = []
        
        border_color = "#28a745" if is_positive else "#dc3545"
        bg_color = "#f8f9fa"
        
        with st.container():
            st.markdown(
                f"""
                <div style="border: 2px solid {border_color}; border-radius: 10px; padding: 15px; margin-bottom: 20px; background-color: {bg_color}; color: black;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin-top: 0; color: #333;">#{i+1}: {ctx.get('passage_id', 'Unknown ID')}</h4>
                        <span style="background-color: {border_color}; color: white; padding: 4px 10px; border-radius: 5px; font-weight: bold;">
                            {"POSITIVE" if is_positive else "NEGATIVE"}
                        </span>
                    </div>
                    <p style="font-size: 0.9em; color: #666;">Score: {ctx.get('score', 'N/A')}</p>
                    <hr style="border: 0.5px solid #ccc; margin: 10px 0;">
                    <div style="font-family: serif; line-height: 1.6; color: #111;">
                        {highlight_text(ctx.get('text', ''), keys_data, color_mapping)}
                    </div>
                    <div style="margin-top: 10px; font-size: 0.8em; color: #888;">
                        <strong>Keys matched:</strong> {keys_data}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
