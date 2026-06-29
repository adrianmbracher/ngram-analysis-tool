# IR N-Gram Analysis Tool

A Streamlit-based visualization tool for analyzing and comparing Information Retrieval (IR) results, with a focus on n-gram/key match visualization and performance metrics.

[Live Demo](https://ngram-analysis-tool.streamlit.app/)

## 🚀 Features

- **Performance Metrics:** Automatically calculates Hits@1, Hits@5, Hits@10, and Hits@100 for your datasets.
- **Interactive Visualization:** Browse through queries and their retrieved documents with rich HTML highlighting of matched keys.
- **Smart Highlighting:**
    - <span style="background-color: #28a745; color: white; padding: 2px 4px; border-radius: 4px;">Green</span>: Keys found in positive contexts.
    - <span style="background-color: #dc3545; color: white; padding: 2px 4px; border-radius: 4px;">Red</span>: Keys found in negative contexts.
    - <span style="background-color: #ffc107; color: black; padding: 2px 4px; border-radius: 4px;">Amber</span>: Keys found in both positive and negative contexts.
- **Comparison Mode:** Side-by-side comparison of two different algorithms or experimental runs.
- **Search & Filter:** Quickly find specific questions or filter by whether they have a "top positive" result.
- **Score Analysis:** Hover over highlighted terms to see their relevance scores.

## 🛠 Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ngram-analysis-tool
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install streamlit
    ```

## 📖 Usage

Run the tool using Streamlit:

```bash
streamlit run visualize.py [DATA_PATH]
```

Or set the `NGRAM_DATA_PATH` environment variable:

```bash
export NGRAM_DATA_PATH=/path/to/your/data
streamlit run visualize.py
```

### Data Structure

The tool expects a directory structure like this:

```text
DATA_PATH/
├── DatasetName1/
│   ├── Algorithm1/
│   │   └── results.json
│   └── Algorithm2/
│       └── results.json
└── DatasetName2/
    └── ...
```

### JSON Format

Each JSON file should contain a list of objects with the following structure:

```json
[
  {
    "question": "What is the capital of France?",
    "answers": ["Paris"],
    "positive_ctxs": [
      {
        "passage_id": "doc1",
        "text": "Paris is the capital and most populous city of France."
      }
    ],
    "ctxs": [
      {
        "passage_id": "doc1",
        "score": 1.5,
        "text": "Paris is the capital and most populous city of France.",
        "keys": [
          ["Paris", 0, 1.0],
          ["capital", 13, 0.8]
        ]
      },
      {
        "passage_id": "doc2",
        "score": 0.5,
        "text": "Lyon is a city in France.",
        "keys": [
          ["France", 15, 0.5]
        ]
      }
    ]
  }
]
```

- `keys`: A list where each element is `[text, start_index, score]` or just `text`.

## 🖥 UI Overview

- **Sidebar:** Select your dataset, algorithm, and specific result file. Enable "Comparison Mode" to compare two files.
- **Global Metrics:** Shows aggregate performance (Hits@k) across the entire selected file.
- **Question Selection:** Browse through questions. A checkmark ✅ indicates the top retrieved document is a positive match.
- **Document View:** Shows retrieved documents with highlighting. Documents are bordered in green if they are positive matches, and red otherwise.
