# 🧠 PathMind

An ML-assisted graph path prediction system that combines **Machine Learning** with **Breadth-First Search (BFS)** to predict and evaluate shortest-path decisions in graphs.

Built with **Python**, **scikit-learn**, **Pandas**, and **Joblib**, PathMind generates graph-based training data, performs feature engineering, trains a Random Forest classifier, and compares ML-predicted paths against BFS shortest paths.

---

# Features

* 🧠 Machine-learning-assisted path prediction
* 🌐 Graph-based pathfinding
* 🔎 Breadth-First Search (BFS)
* 🤖 Random Forest classification
* 📊 Automated training dataset generation
* ⚙️ Graph feature engineering
* 🎯 Candidate-node classification
* 🛣 Shortest-path prediction
* 🔄 ML path vs BFS path comparison
* 📈 Accuracy evaluation
* 📋 Precision, recall, and F1-score evaluation
* 📉 Confusion matrix generation
* 💾 Trained model persistence using Joblib
* ⌨️ Interactive graph input through PowerShell
* 🧪 Independent BFS testing
* 🧪 Feature-engineering testing
* 🧩 Modular Python architecture
* 🌍 Cross-platform Python implementation

---

# How It Works

PathMind uses BFS as the algorithmic reference and machine learning to learn which candidate nodes are likely to contribute to a shortest path.

```text
                    Graph
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
                  Comparison
                      │
                      ▼
                  Final Result
```

---

# Requirements

* Python 3.10 or later
* pip
* PowerShell / Terminal

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

```powershell
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

---

# Generate the Training Dataset

Generate graph-based training data:

```powershell
python -m src.dataset_generator
```

The generated dataset is stored in:

```text
data/
└── training_data.csv
```

Check the generated dataset:

```powershell
Get-ChildItem .\data
```

View the first few rows:

```powershell
Get-Content .\data\training_data.csv -TotalCount 6
```

---

# Train the Machine-Learning Model

Train the Random Forest model:

```powershell
python -m src.model
```

The training process:

1. Loads the training dataset
2. Separates features and labels
3. Splits the dataset into training and testing sets
4. Creates a Random Forest classifier
5. Trains the model
6. Evaluates model performance
7. Saves the trained model

The trained model is saved to:

```text
models/
└── bfs_shortest_path_model.joblib
```

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

The current Random Forest model achieved:

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

> Classification accuracy represents candidate-node classification performance. It should not be interpreted as complete shortest-path accuracy. Path-level performance is evaluated separately by comparing ML-generated routes with BFS routes.

---

# Feature Engineering

PathMind currently extracts **16 features** for each candidate node.

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

These features describe the relationship between the source, destination, current node, candidate node, and graph structure.

---

# Interactive Prediction

After training the model, run:

```powershell
python -m src.predictor
```

PathMind will request the graph information interactively.

Example:

```text
============================================================
                 GRAPH INPUT
============================================================

Enter number of nodes: 8

Nodes are numbered from 0 to 7.

Enter the number of edges.
Number of edges: 10

Enter each edge as: node1 node2

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

PathMind then calculates the BFS shortest path and the ML-predicted path.

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

# BFS Testing

The BFS implementation can be tested independently:

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

# Feature Engineering Testing

Test the feature-engineering module:

```powershell
python .\src\feature_engineering.py
```

The test displays the generated features, feature count, feature vector, and candidate label.

Example:

```text
FEATURE ENGINEERING TEST

============================================================

Testing candidate:
Source      : 0
Destination : 6
Current     : 0
Candidate   : 2

Feature count:
16

Candidate label:
1

# Feature engineering test completed successfully.
```

---

# Syntax Checking

Check individual Python modules before running them:

```powershell
python -m py_compile .\src\bfs.py
python -m py_compile .\src\feature_engineering.py
python -m py_compile .\src\dataset_generator.py
python -m py_compile .\src\model.py
python -m py_compile .\src\predictor.py
```

No output generally indicates that the file passed Python's compilation check.

---

# Complete Workflow

For a complete training and prediction workflow:

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Generate training data
python -m src.dataset_generator

# Train the ML model
python -m src.model

# Run interactive prediction
python -m src.predictor
```

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
│   ├── graph_generator.py
│   ├── feature_engineering.py
│   ├── dataset_generator.py
│   ├── model.py
│   └── predictor.py
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# Module Description

| File                     | Description                                                                    |
| ------------------------ | ------------------------------------------------------------------------------ |
| `main.py`                | Main project entry point                                                       |
| `bfs.py`                 | BFS traversal, shortest-path search, distance calculation, and path validation |
| `graph_generator.py`     | Graph generation utilities                                                     |
| `feature_engineering.py` | Generates ML features from graph information                                   |
| `dataset_generator.py`   | Creates the supervised-learning dataset                                        |
| `model.py`               | Trains and evaluates the Random Forest classifier                              |
| `predictor.py`           | Loads the trained model and predicts paths                                     |
| `__init__.py`            | Initializes the Python package                                                 |

---

# Machine Learning Pipeline

```text
Graph Generation
       │
       ▼
BFS Ground Truth
       │
       ▼
Feature Extraction
       │
       ▼
Training Dataset
       │
       ▼
Train/Test Split
       │
       ▼
Random Forest
       │
       ▼
Model Evaluation
       │
       ▼
Save Model
       │
       ▼
User Graph
       │
       ▼
ML Path Prediction
       │
       ▼
BFS Comparison
```

---

# Technologies

| Technology    | Purpose               |
| ------------- | --------------------- |
| Python        | Core implementation   |
| Pandas        | Dataset processing    |
| scikit-learn  | Machine learning      |
| Random Forest | Classification model  |
| Joblib        | Model persistence     |
| Matplotlib    | Visualization         |
| PowerShell    | Interactive execution |
| Git           | Version control       |
| GitHub        | Repository hosting    |

---

# Limitations

The current version is a research/educational prototype.

* The model is trained using BFS-derived data.
* ML predictions are not guaranteed to be optimal.
* BFS remains the ground-truth reference for unweighted graphs.
* Classification accuracy does not directly represent path-level accuracy.
* The current implementation focuses on unweighted graphs.
* Performance may vary on previously unseen graph structures.
* Larger and more diverse datasets may improve generalization.
* The current model does not replace traditional shortest-path algorithms.

---

# Future Improvements

Possible future improvements include:

* 📊 Large-scale graph dataset generation
* 🎯 Automated path-level accuracy testing
* 🧪 Evaluation across hundreds or thousands of unseen graphs
* ⚖️ Weighted graph support
* ➡️ Directed graph support
* 🧭 Dijkstra-based path prediction
* ⭐ A* path prediction
* 🧠 Graph Neural Networks (GNNs)
* 🔬 Hyperparameter optimization
* 🔄 Cross-validation
* 📈 Advanced model comparison
* 🖥 Graphical user interface
* 🌐 Web-based visualization
* 📊 Real-time graph visualization
* 🔍 Explainable AI for node selection
* ⚡ Performance benchmarking against BFS, Dijkstra, and A*

---

# Testing Strategy

PathMind can be tested at multiple levels:

### Unit Testing

Test individual components:

```text
BFS
Feature Engineering
Graph Generation
Path Validation
```

### Model Testing

Evaluate:

```text
Accuracy
Precision
Recall
F1-score
Confusion Matrix
```

### Path-Level Testing

Compare:

```text
ML predicted path
        vs.
BFS shortest path
```

Important path-level metrics can include:

* Exact path match
* Destination reachability
* Path length
* Optimal-path rate
* Average path-length difference

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
5. Verify the project
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
* **Python** for the core implementation

---

# Author

**Fuad Sanin**

GitHub: https://github.com/Sanin1024

---

# Project Repository

GitHub:

https://github.com/Sanin1024/PathMind
