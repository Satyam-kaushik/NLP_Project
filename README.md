#  Emotion-Cause Pair Detection using GCN on RECCON Dataset

## 📌 Overview
This project tackles the task of **emotion-cause clause pair detection** in **multi-turn conversations** using the **RECCON dataset**. It segments conversations into semantically meaningful clauses, labels them as emotion or cause, builds a graph of clauses, and uses a **Graph Convolutional Network (GCN)** to detect relationships between emotions and their causes.

---

## ✅ Tasks

### 🔹 Task 1: Dataset Preparation

- **Input**: RECCON-style conversation files (`tr_xxxx.json`), where each turn contains utterances and associated emotion/cause labels.
- **Goal**: Break each utterance into meaningful **clauses** for further processing.

#### 🔍 Clause Segmentation Approach

- Split utterances using punctuation: `.`, `!`, `?`, `,`.
- Use **coordinating conjunctions** (`and`, `but`, `or`) and **subordinating conjunctions** (`because`, `although`, `if`, etc.).
- Ensure that **each clause forms a semantically independent thought**.
- Implemented using **spaCy dependency parsing** for robust, reproducible clause boundaries.

---

### 🔹 Task 2: Clause Annotation

#### 🏷️ Objective

Automatically label each clause as:
- **Emotion Clause**: Expresses emotion.
- **Cause Clause**: Explains the reason for an emotion.

#### 🧠 Annotation Strategy

- For **non-neutral utterances**, mark **all clauses** as **emotion clauses**.
- To identify **cause clauses**:
  - Compute **semantic similarity** between clause embeddings and the "expanded emotion cause span" from RECCON annotations.
  - Use **Sentence-BERT** embeddings for similarity computation.
  - A clause is labeled as a **cause** if its similarity to a cause span exceeds a defined threshold.

---

### 🔹 Task 3: Model - Graph Construction & GCN

#### 🧩 Clause Graph Construction

- Each **clause** is a **node**.
- Add **edges** based on:
  - **Syntactic dependencies**: subject-verb, verb-object relations using dependency parse tree.
  - **Temporal proximity**: clauses in the same or nearby utterances.

#### 🔡 Embedding Representation

- Use **Sentence-BERT** or **RoBERTa** for high-quality clause embeddings.

#### 🧠 Graph Neural Network

- Employ a **2-layer GCN** to learn from the graph:
  - **Input**: Clause embeddings
  - **Layers**: GCNConv → ReLU → GCNConv
  - **Output**: Node classification — **emotion**, **cause**, or **none**

- A post-processing step identifies **emotion-cause pairs** using attention/similarity-based pairing.

---

## 📊 Evaluation Metrics

- **Precision**, **Recall**, **F1-Score** for Emotion-Cause Pair detection.
- **AUC-ROC** to assess discriminative power.
- **Confusion Matrix** interpretation:
  - True Positives: Correct clause pairs
  - False Positives/Negatives: Misclassified or missed causes/emotions

---

## 🛠️ Tools & Libraries

| Component              | Tool/Library         |
|------------------------|----------------------|
| Programming Language   | Python               |
| Clause Segmentation    | spaCy (en_core_web_sm) |
| Embeddings             | Sentence-BERT        |
| Graph Construction     | NetworkX             |
| Graph Model            | PyTorch Geometric    |
| Deep Learning          | PyTorch              |
| Visualization          | Matplotlib           |

---


