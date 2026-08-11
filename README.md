# 🧠 PathMind

An ML-assisted graph path prediction system that combines **Machine Learning** with **Breadth-First Search (BFS)** to predict and evaluate shortest-path decisions in graphs.

Built with **Python**, **scikit-learn**, **Pandas**, **Joblib**, and **Matplotlib**, PathMind generates graph-based training data, performs feature engineering, trains a Random Forest classifier, predicts graph paths, and compares machine-learning decisions against BFS shortest paths.

---

# Features

* 🧠 Machine-learning-assisted graph path prediction
* 🔎 Breadth-First Search (BFS) shortest-path reference
* 🤖 Random Forest classification
* 🌐 Graph generation and processing
* 📊 Automated training dataset generation
* ⚙️ Graph feature engineering
* 🎯 Candidate-node classification
* 🛣 ML-based path prediction
* 🔄 ML vs BFS path comparison
* 📈 Model accuracy evaluation
* 📋 Precision, recall, and F1-score
* 📉 Confusion matrix evaluation
* 💾 Trained model persistence using Joblib
* 📊 Graph and path visualization
* ⌨️ Interactive graph input
* 🧪 BFS testing
* 🧪 Feature-engineering testing
* 🧪 Model evaluation
* 🧩 Modular Python architecture
* 🌍 Cross-platform Python implementation

---

# How It Works

PathMind uses **BFS-generated shortest-path information as the reference** for supervised learning.

The system extracts graph features for candidate nodes and trains a Random Forest classifier to identify promising path-selection candidates.

```text
                         Graph
                           │
                           ▼
                  Graph Generation
                           │
                           ▼
                  BFS Ground Truth
                           │
                           ▼
                 Feature Engineering
                           │
                           ▼
                    Training Data
                           │
                           ▼
                  Random Forest Model
                           │
                           ▼
                    Trained Model
                           │
                           ▼
                   User Graph Input
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
                 BFS             ML Model
                  │                 │
                  ▼                 ▼
            Shortest Path     Predicted Path
                  │                 │
                  └────────┬────────┘
                           ▼
                       Evaluation
                           │
                           ▼
                  PathMind Results
```

---

# Requirements

* Python 3.10 or later
* pip
* Windows PowerShell, macOS Terminal, or Linux shell

Python dependencies:

* pandas
* scikit-learn
* joblib
* matplotlib

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Sanin1024/PathMind.git
cd PathMind
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Quick Start

For the complete PathMind workflow:

```powershell
python train.py
python predict.py
python evaluate.py
```

The general workflow is:

```text
Train
  ↓
Generate / Load Dataset
  ↓
Train Random Forest
  ↓
Save Model
  ↓
Predict
  ↓
Enter Graph
  ↓
Generate ML Path
  ↓
Compare with BFS
  ↓
Evaluate
```

---

# Training

Train the machine-learning model using:

```powershell
python train.py
```

The training process uses graph-derived features and BFS-based path information to train a Random Forest classifier.

The trained model is stored in:

```text
models/
└── bfs_shortest_path_model.joblib
```

---

# Generate Training Data

PathMind includes a dataset-generation module for creating graph-based training examples.

Run:

```powershell
python -m src.dataset_generator
```

The generated dataset is stored in:

```text
data/
└── training_data.csv
```

The dataset contains graph features and candidate labels used by the machine-learning model.

---

# Model

PathMind currently uses a:

**Random Forest Classifier**

The model learns from graph features describing relationships between:

* Source node
* Destination node
* Current node
* Candidate node
* Graph distances
* Node degrees
* Neighbor relationships
* Visited-node status
* Reachability

The model's output is used to assist with path-selection decisions.

---

# Features Used by the Model

PathMind currently uses **16 engineered features**:

|  # | Feature                                |
| -: | -------------------------------------- |
|  1 | `source`                               |
|  2 | `destination`                          |
|  3 | `current_node`                         |
|  4 | `candidate_node`                       |
|  5 | `source_candidate_distance`            |
|  6 | `candidate_destination_distance`       |
|  7 | `current_destination_distance`         |
|  8 | `current_degree`                       |
|  9 | `candidate_degree`                     |
| 10 | `destination_degree`                   |
| 11 | `candidate_is_destination`             |
| 12 | `candidate_is_neighbor_of_destination` |
| 13 | `candidate_is_visited`                 |
| 14 | `candidate_is_source`                  |
| 15 | `candidate_distance_from_current`      |
| 16 | `destination_reachable_from_candidate` |

These features provide the model with information about the current graph state and the relationship between candidate nodes and the target destination.

---

# Model Performance

The current training dataset contains:

```text
Rows    : 1075
Columns : 18
```

Class distribution:

```text
Label 0 : 879
Label 1 : 196
```

Training/testing split:

```text
Training samples : 860
Testing samples  : 215
```

The current Random Forest training run achieved:

```text
Accuracy: 0.9395
```

Classification report:

```text
              precision    recall  f1-score   support

           0       0.98      0.94      0.96       176
           1       0.78      0.92      0.85        39

    accuracy                           0.94       215
   macro avg       0.88      0.93      0.90       215
weighted avg       0.95      0.94      0.94       215
```

Confusion matrix:

```text
[[166  10]
 [  3  36]]
```

> **Note:** The 93.95% accuracy represents candidate-node classification performance. It should not be interpreted as complete shortest-path accuracy. Path-level performance should be evaluated separately by comparing ML-generated paths against BFS shortest paths.

---

# Prediction

Use the interactive prediction interface:

```powershell
python predict.py
```

The user can provide the graph and specify the source and destination nodes.

Example:

```text
Enter number of nodes: 8

Enter number of edges: 10

Edge 1/10: 0 1
Edge 2/10: 0 3
Edge 3/10: 1 2
Edge 4/10: 1 4
Edge 5/10: 2 5
Edge 6/10: 3 4
Edge 7/10: 3 6
Edge 8/10: 4 5
Edge 9/10: 5 7
Edge 10/10: 6 7

Enter source node: 0
Enter destination node: 7
```

PathMind then evaluates the graph using both BFS and the trained ML model.

Example:

```text
BFS shortest path : 0 -> 3 -> 6 -> 7
ML predicted path : 0 -> 3 -> 6 -> 7

BFS path length   : 3
ML path length    : 3

Same path          : True
Reached destination: True
```

---

# BFS

PathMind contains a complete BFS implementation in:

```text
src/bfs.py
```

It provides:

* BFS traversal
* Shortest-path search
* Distance calculation
* Parent tracking
* Path reconstruction
* Path validation
* Path formatting
* BFS tree generation

Test the BFS implementation directly:

```powershell
python .\src\bfs.py
```

Example:

```text
Testing BFS...

Graph:
0: [1, 2]
1: [0, 3, 4]
2: [0, 5]
3: [1]
4: [1, 5]
5: [2, 4, 6]
6: [5]

BFS Traversal:
0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6

Shortest Path:
0 -> 2 -> 5 -> 6

Shortest Distance:
3

Path Valid:
True

BFS test completed successfully.
```

---

# Evaluation

PathMind includes a dedicated evaluation script:

```powershell
python evaluate.py
```

The evaluation component can be used to assess model/path performance beyond simple classification accuracy.

The key concept is comparing:

```text
ML Predicted Path
        vs.
BFS Shortest Path
```

Important path-level metrics include:

* Exact path match
* Destination reachability
* Predicted path length
* BFS path length
* Optimal-path performance
* Path-length difference

---

# Visualization

PathMind includes graph visualization functionality in:

```text
src/visualize.py
```

The visualization component uses **Matplotlib** to support graphical inspection of graphs and paths.

Visualization can be used to understand:

* Graph structure
* Source node
* Destination node
* Candidate paths
* BFS path
* ML-predicted path

This provides a visual way to inspect how PathMind navigates a graph.

---

# Feature Engineering

Test the feature-engineering module directly:

```powershell
python .\src\feature_engineering.py
```

Example:

```text
FEATURE ENGINEERING TEST

============================================================

Testing candidate:
Source      : 0
Destination : 6
Current     : 0
Candidate   : 2

Features:
source: 0.0
destination: 6.0
current_node: 0.0
candidate_node: 2.0
source_candidate_distance: 1.0
candidate_destination_distance: 2.0
current_destination_distance: 3.0
current_degree: 2.0
candidate_degree: 2.0
destination_degree: 1.0
candidate_is_destination: 0.0
candidate_is_neighbor_of_destination: 0.0
candidate_is_visited: 0.0
candidate_is_source: 0.0
candidate_distance_from_current: 1.0
destination_reachable_from_candidate: 1.0

Feature count:
16

Candidate label:
1
```

---

# Python Module Interface

The core modules can also be executed directly.

Generate dataset:

```powershell
python -m src.dataset_generator
```

Train model:

```powershell
python -m src.model
```

Run prediction:

```powershell
python -m src.predictor
```

Test BFS:

```powershell
python .\src\bfs.py
```

Test feature engineering:

```powershell
python .\src\feature_engineering.py
```

---

# Syntax Checking

Python modules can be checked before execution using:

```powershell
python -m py_compile .\src\bfs.py
python -m py_compile .\src\feature_engineering.py
python -m py_compile .\src\dataset_generator.py
python -m py_compile .\src\model.py
python -m py_compile .\src\predictor.py
```

No output generally indicates that the module passed Python's compilation check.

---

# Project Structure

```text
PathMind/
│
├── data/
│   └── training_data.csv
│
├── models/
│   └── bfs_shortest_path_model.joblib
│
├── src/
│   ├── __init__.py
│   ├── bfs.py
│   ├── dataset_generator.py
│   ├── feature_engineering.py
│   ├── graph_generator.py
│   ├── model.py
│   ├── predictor.py
│   └── visualize.py
│
├── .gitignore
├── README.md
├── LICENSE
├── main.py
├── train.py
├── predict.py
├── evaluate.py
└── requirements.txt
```

---

# Project Components

| Component                               | Purpose                                     |
| --------------------------------------- | ------------------------------------------- |
| `main.py`                               | Main application entry point                |
| `train.py`                              | Training entry point                        |
| `predict.py`                            | Prediction entry point                      |
| `evaluate.py`                           | Evaluation entry point                      |
| `src/bfs.py`                            | BFS traversal and shortest-path operations  |
| `src/graph_generator.py`                | Graph generation utilities                  |
| `src/feature_engineering.py`            | Graph feature extraction                    |
| `src/dataset_generator.py`              | ML dataset generation                       |
| `src/model.py`                          | Random Forest training and model evaluation |
| `src/predictor.py`                      | ML-based path prediction                    |
| `src/visualize.py`                      | Graph and path visualization                |
| `data/training_data.csv`                | Training dataset                            |
| `models/bfs_shortest_path_model.joblib` | Saved trained model                         |

---

# Complete Workflow

```text
                 ┌───────────────────┐
                 │  Graph Generator  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │       BFS         │
                 │  Ground Truth     │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Feature           │
                 │ Engineering       │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Training Dataset  │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Random Forest     │
                 │ Training          │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Trained Model     │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ User Graph Input   │
                 └─────────┬─────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          ┌─────────────┐      ┌─────────────┐
          │     BFS     │      │ ML Predictor│
          └──────┬──────┘      └──────┬──────┘
                 │                    │
                 ▼                    ▼
          BFS Shortest Path     ML Predicted Path
                 │                    │
                 └─────────┬──────────┘
                           ▼
                  ┌─────────────────┐
                  │   Evaluation    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Visualization   │
                  └─────────────────┘
```

---

# Testing

PathMind can be tested at several levels.

### BFS Testing

```powershell
python .\src\bfs.py
```

### Feature Engineering Testing

```powershell
python .\src\feature_engineering.py
```

### Dataset Generation

```powershell
python -m src.dataset_generator
```

### Model Training

```powershell
python train.py
```

### Prediction

```powershell
python predict.py
```

### Evaluation

```powershell
python evaluate.py
```

---

# Limitations

PathMind is currently an **ML-assisted graph path prediction prototype**.

* BFS provides the shortest-path reference for unweighted graphs.
* ML predictions are not guaranteed to produce an optimal path.
* Classification accuracy does not directly represent complete path accuracy.
* The current approach is based on BFS-derived training data.
* Performance can vary across unseen graph structures.
* Larger and more diverse datasets may improve generalization.
* The current implementation focuses primarily on unweighted graph navigation.
* Traditional graph algorithms remain important when guaranteed optimality is required.

---

# Future Improvements

Possible future improvements include:

* 📊 Larger and more diverse graph datasets
* 🎯 Automated path-level accuracy benchmarking
* 🧪 Evaluation across thousands of unseen graphs
* ⚖️ Weighted graph support
* ➡️ Directed graph support
* 🚀 Dijkstra-based path prediction
* ⭐ A* path prediction
* 🧠 Graph Neural Network implementation
* 🔬 Hyperparameter optimization
* 🔄 Cross-validation
* 📈 Model comparison
* 🖥 Interactive graphical interface
* 🌐 Web-based graph visualization
* ⚡ Performance benchmarking
* 🔍 Explainable AI for path decisions
* 📊 Real-time ML vs algorithm comparison

---

# Technologies

| Technology    | Purpose                   |
| ------------- | ------------------------- |
| Python        | Core programming language |
| Pandas        | Dataset processing        |
| scikit-learn  | Machine learning          |
| Random Forest | Classification            |
| Joblib        | Model serialization       |
| Matplotlib    | Graph visualization       |
| PowerShell    | Command-line interaction  |
| Git           | Version control           |
| GitHub        | Repository hosting        |

---

# Dependencies

Runtime:

```text
pandas
scikit-learn
joblib
matplotlib
```

Install all dependencies using:

```powershell
pip install -r requirements.txt
```

---

# Why PathMind?

Traditional BFS can reliably find the shortest path in an unweighted graph.

PathMind explores a different question:

> **Can machine learning learn graph-based path-selection patterns from algorithmically generated data?**

BFS acts as the reference mechanism for generating path information, while the machine-learning model learns from engineered graph features and attempts to make intelligent candidate-node decisions.

This makes PathMind an experimental intersection of:

```text
Graph Algorithms
       +
Feature Engineering
       +
Supervised Machine Learning
       +
Path Prediction
       +
Algorithmic Evaluation
```

---

# License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

---

# Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add or update tests
5. Run the project and verify your changes
6. Commit your changes
7. Push the branch
8. Open a Pull Request

Example:

```bash
git checkout -b feature/new-improvement
```

---

# Acknowledgements

* **scikit-learn** for machine-learning algorithms
* **Pandas** for data processing
* **Joblib** for model persistence
* **Matplotlib** for visualization
* **Python** for the core implementation

---

# Author

**Fuad Sanin**

GitHub: https://github.com/Sanin1024

---

# Repository

**PathMind**

https://github.com/Sanin1024/PathMind
