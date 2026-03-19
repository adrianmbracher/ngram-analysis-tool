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
    
    # File 1 Selection
    st.sidebar.subheader("File 1 (Base)")
    ds_path = os.path.join('.', selected_dataset)
    algorithms = sorted([d for d in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, d))])
    
    if not algorithms:
        st.sidebar.warning(f"No algorithm directories found in {selected_dataset}.")
        return
        
    default_algo = "BEAM"
    algo_idx = algorithms.index(default_algo) if default_algo in algorithms else 0
    selected_algo_1 = st.sidebar.selectbox("Algorithm 1", algorithms, index=algo_idx, key="algo1")
    
    algo_path_1 = os.path.join(ds_path, selected_algo_1)
    json_files_1 = sorted([f for f in os.listdir(algo_path_1) if f.endswith('.json')])
    
    if not json_files_1:
        st.sidebar.warning(f"No JSON files found in {selected_algo_1}.")
        return
        
    default_file = "output_seal_limit_beam_intersect.json"
    file_idx_1 = json_files_1.index(default_file) if default_file in json_files_1 else 0
    selected_file_1 = st.sidebar.selectbox("File 1", json_files_1, index=file_idx_1, key="file1")
    file_path_1 = os.path.join(algo_path_1, selected_file_1)

    # Comparison Mode
    comparison_mode = st.sidebar.checkbox("Comparison Mode")
    file_path_2 = None
    data_2 = None
    
    if comparison_mode:
        st.sidebar.subheader("File 2 (Comparison)")
        selected_algo_2 = st.sidebar.selectbox("Algorithm 2", algorithms, index=algo_idx, key="algo2")
        algo_path_2 = os.path.join(ds_path, selected_algo_2)
        json_files_2 = sorted([f for f in os.listdir(algo_path_2) if f.endswith('.json')])
        
        if not json_files_2:
            st.sidebar.warning(f"No JSON files found in {selected_algo_2}.")
        else:
            # Try to match the filename from file 1 if possible
            file_idx_2 = json_files_2.index(selected_file_1) if selected_file_1 in json_files_2 else 0
            selected_file_2 = st.sidebar.selectbox("File 2", json_files_2, index=file_idx_2, key="file2")
            file_path_2 = os.path.join(algo_path_2, selected_file_2)

    try:
        data_1 = load_data(file_path_1)
        if file_path_2:
            data_2 = load_data(file_path_2)
            # Create a mapping for quick lookup by question
            data_2_map = {item['question']: item for item in data_2}
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return

    # Calculate Metrics
    hits_metrics_1 = calculate_hits_at_k(file_path_1, ks=(1, 5, 10, 100))
    if file_path_2:
        hits_metrics_2 = calculate_hits_at_k(file_path_2, ks=(1, 5, 10, 100))
    
    # Display Metrics at the top
    if not comparison_mode:
        st.subheader(f"📊 Global Metrics: {selected_file_1}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Queries", len(data_1))
        m2.metric("Hits@1", f"{hits_metrics_1[1]:.1f}%")
        m3.metric("Hits@5", f"{hits_metrics_1[5]:.1f}%")
        m4.metric("Hits@10", f"{hits_metrics_1[10]:.1f}%")
        m5.metric("Hits@100", f"{hits_metrics_1[100]:.1f}%")
    else:
        st.subheader(f"📊 Comparison: {selected_file_1} vs {selected_file_2}")
        cols = st.columns(5)
        cols[0].write("**Metric**")
        cols[0].write("Total Queries")
        cols[0].write("Hits@1")
        cols[0].write("Hits@5")
        cols[0].write("Hits@10")
        cols[0].write("Hits@100")
        
        for i, (name, metrics) in enumerate([("File 1", hits_metrics_1), ("File 2", hits_metrics_2)]):
            c = cols[i+1]
            c.write(f"**{name}**")
            c.write(f"{len(data_1) if i==0 else len(data_2)}")
            for k in [1, 5, 10, 100]:
                val = metrics[k]
                c.write(f"{val:.1f}%")

    st.divider()

    # Search and Filter
    query_search = st.sidebar.text_input("Search Question")
    filtered_data = [d for d in data_1 if query_search.lower() in d['question'].lower()]
    
    if not filtered_data:
        st.warning("No queries match the search criteria.")
        return

    def has_top_positive(item):
        p_ids = {p.get('passage_id') or p.get('text', '') for p in item.get('positive_ctxs', [])}
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
    
    item_1 = filtered_data[selected_idx]
    item_2 = data_2_map.get(item_1['question']) if comparison_mode else None
    
    def get_key_data(item):
        if not item: return {}, {}, set()
        positive_ids = {p.get('passage_id') or p.get('text', '') for p in item.get('positive_ctxs', [])}
        pos_keys = {}
        neg_keys = {}
        for ctx in item.get('ctxs', []):
            is_pos = (ctx.get('passage_id') or ctx.get('text', '')) in positive_ids
            keys_data = ctx.get('keys', [])
            if isinstance(keys_data, str):
                try: keys_data = json.loads(keys_data)
                except: keys_data = []
            for k in keys_data:
                key_text = str(k[0]).strip() if isinstance(k, list) and k else k.strip() if isinstance(k, str) else ""
                if key_text:
                    target = pos_keys if is_pos else neg_keys
                    score = k[2] if isinstance(k, list) and len(k) >= 3 else 0.0
                    if key_text not in target or score > target[key_text]:
                        target[key_text] = score
        
        color_mapping = {}
        all_keys = {}
        unique_texts = set(pos_keys.keys()) | set(neg_keys.keys())
        for k in unique_texts:
            is_in_pos = k in pos_keys
            is_in_neg = k in neg_keys
            max_score = max(pos_keys.get(k, 0.0), neg_keys.get(k, 0.0))
            if is_in_pos and is_in_neg:
                color_mapping[k.lower()] = {"bg": "#ffc107", "fg": "black"}
            elif is_in_pos:
                color_mapping[k.lower()] = {"bg": "#28a745", "fg": "white"}
            else:
                color_mapping[k.lower()] = {"bg": "#dc3545", "fg": "white"}
            all_keys[k] = max_score
        return all_keys, color_mapping, positive_ids

    keys_1, colors_1, p_ids_1 = get_key_data(item_1)
    keys_2, colors_2, p_ids_2 = get_key_data(item_2) if item_2 else ({}, {}, set())

    st.header(f"Question: {item_1['question']}")
    
    # Comparison display
    if not comparison_mode:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.subheader("Expected Answers")
            for ans in item_1.get('answers', []): st.markdown(f"- {ans}")
        with col2:
            st.subheader("Stats")
            st.write(f"Total retrieved: {len(item_1.get('ctxs', []))}")
            st.write(f"Total positives: {len(item_1.get('positive_ctxs', []))}")
        with col3:
            st.subheader("All Keys Matched")
            sorted_keys = sorted(keys_1.items(), key=lambda x: x[1], reverse=True)
            badges = [f'<span title="Max Score: {s:.2f}" style="background-color: {colors_1[k.lower()]["bg"]}; color: {colors_1[k.lower()]["fg"]}; padding: 2px 6px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 4px; cursor: help;">{k}</span>' for k, s in sorted_keys]
            st.markdown(f'<div>{" ".join(badges) if badges else "None"}</div>', unsafe_allow_html=True)
    else:
        st.subheader("Key Comparison")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Only in File 1**")
            diff1 = set(keys_1.keys()) - set(keys_2.keys())
            sorted_diff1 = sorted([(k, keys_1[k]) for k in diff1], key=lambda x: x[1], reverse=True)
            badges = [f'<span title="Score: {s:.2f}" style="background-color: {colors_1[k.lower()]["bg"]}; color: {colors_1[k.lower()]["fg"]}; padding: 2px 6px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 4px; cursor: help;">{k}</span>' for k, s in sorted_diff1]
            st.markdown(f'<div>{" ".join(badges) if badges else "None"}</div>', unsafe_allow_html=True)
        with c2:
            st.write("**Common Keys**")
            common = set(keys_1.keys()) & set(keys_2.keys())
            sorted_common = sorted([(k, max(keys_1[k], keys_2[k])) for k in common], key=lambda x: x[1], reverse=True)
            badges = [f'<span title="Max Score: {s:.2f}" style="background-color: {colors_1[k.lower()]["bg"]}; color: {colors_1[k.lower()]["fg"]}; padding: 2px 6px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 4px; cursor: help;">{k}</span>' for k, s in sorted_common]
            st.markdown(f'<div>{" ".join(badges) if badges else "None"}</div>', unsafe_allow_html=True)
        with c3:
            st.write("**Only in File 2**")
            diff2 = set(keys_2.keys()) - set(keys_1.keys())
            sorted_diff2 = sorted([(k, keys_2[k]) for k in diff2], key=lambda x: x[1], reverse=True)
            badges = [f'<span title="Score: {s:.2f}" style="background-color: {colors_2[k.lower()]["bg"]}; color: {colors_2[k.lower()]["fg"]}; padding: 2px 6px; border-radius: 4px; margin-right: 4px; display: inline-block; margin-bottom: 4px; cursor: help;">{k}</span>' for k, s in sorted_diff2]
            st.markdown(f'<div>{" ".join(badges) if badges else "None"}</div>', unsafe_allow_html=True)

    st.divider()

    # Document Display
    if not comparison_mode:
        st.subheader("Retrieved Documents")
        for i, ctx in enumerate(item_1.get('ctxs', [])):
            display_ctx(i, ctx, p_ids_1, colors_1)
    else:
        if not item_2:
            st.error("Question not found in File 2")
            return
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.subheader(f"File 1: {selected_algo_1}")
            for i, ctx in enumerate(item_1.get('ctxs', [])):
                display_ctx(i, ctx, p_ids_1, colors_1)
        with col_f2:
            st.subheader(f"File 2: {selected_algo_2}")
            for i, ctx in enumerate(item_2.get('ctxs', [])):
                display_ctx(i, ctx, p_ids_2, colors_2)

def display_ctx(i, ctx, positive_ids, color_mapping):
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
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
