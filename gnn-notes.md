# Graph Neural Networks — Learning Notes

This repository contains my personal learning notes on **Graph Neural Networks (GNNs)**.
These notes are based on my complete study of the book  
**Deep Learning on Graphs** by *Yao Ma and Jiliang Tang*.

I created these notes to build strong fundamentals, revise key ideas,
and prepare for industry-level GNN work and interviews.
Everything is written in **my own words** for clarity and understanding.

---

## UNIT 1: Foundations

### 1. Introduction to Learning on Graphs
- Many real-world problems are inherently **relational**, such as social networks,
  transaction systems, molecular structures, and knowledge graphs.
- Traditional machine learning assumes data points are independent (i.i.d.),
  which does not hold for graph-structured data.
- Graphs naturally represent **entities as nodes** and **relationships as edges**.
- The main goal is to learn representations that capture both **node features**
  and **graph structure**.

Why graphs are important:
- Classical ML treats samples in isolation.
- Graph-based learning explicitly models **dependencies and interactions**.

---

### 2. Graph Fundamentals
- A graph is defined as **G = (V, E)**, where:
  - V is the set of nodes
  - E is the set of edges
- Graphs can be directed or undirected, weighted or unweighted.

Key concepts:
- Adjacency matrix (A)
- Degree matrix (D)
- Neighborhood of a node N(v)
- Walks, paths, and connectivity
- Centrality measures:
  - Degree centrality
  - Eigenvector centrality
  - Betweenness centrality

Why this matters for GNNs:
- Neighborhoods define **message passing**
- Degree affects **normalization**
- Paths enable **multi-hop information flow**

---

### 3. Deep Learning Foundations
- Neural networks learn representations using linear transformations
  followed by non-linear activation functions.
- CNNs aggregate local neighborhoods on grid-like data.
- RNNs model dependencies across sequences.
- GNNs generalize these ideas to **arbitrary graph structures**.

Important concepts:
- Feedforward neural networks
- Activation functions (ReLU, sigmoid)
- Backpropagation and optimization
- Overfitting and regularization

---

## UNIT 2: Spectral Graph Theory

### 4. Graph Laplacian
- Unnormalized Laplacian: **L = D − A**
- Symmetric normalized Laplacian:
  **L_sym = I − D^{-1/2} A D^{-1/2}**

Why normalization is needed:
- Prevents high-degree nodes from dominating
- Improves numerical stability
- Ensures balanced information propagation

---

### 5. Graph Fourier Transform
- A graph signal assigns a value (or vector) to each node.
- Eigenvectors of the graph Laplacian act as the **Fourier basis** for graphs.
- Eigenvalues correspond to graph frequencies.

Definitions:
- Forward GFT: **x̂ = Uᵀ x**
- Inverse GFT: **x = U x̂**

Interpretation:
- Low-frequency components represent smooth signals on the graph.
- High-frequency components represent sharp changes across edges.

Connection to GNNs:
- Many GNN layers behave like **low-pass filters**
- Over-smoothing can be understood using this spectral perspective

---

## UNIT 3: Graph Embedding

### 6. Graph Embedding Overview
- Graph embedding maps nodes into low-dimensional vector spaces.
- These embeddings can be used for downstream ML tasks.
- Different methods preserve different graph properties.

---

### 7. Preserving Node Co-occurrence
- Nodes are considered similar if they frequently appear together
  in random walks.
- This idea is inspired by Word2Vec from NLP.

Common methods:
- DeepWalk
- Node2Vec

What is preserved:
- Local proximity
- Neighborhood similarity

Limitations:
- Embeddings are task-agnostic
- Node features are usually ignored

---

### 8. Preserving Structural Role
- Nodes can have similar **structural roles** even if they are far apart.
- Structural equivalence focuses on function rather than proximity.

Method:
- struc2vec

Preserves:
- Node roles
- Structural patterns

---

### 9. Preserving Node Status
- Some nodes are more influential or important than others.
- Importance is not only determined by node degree.

Preserves:
- Centrality
- Hierarchical position
- Influence within the graph

---

### 10. Preserving Community Structure
- Communities are groups of nodes with dense internal connections.
- Nodes within the same community should be close in embedding space.

Preserves:
- Clusters
- Community boundaries

---

### 11. Embedding on Complex Graphs
- Heterogeneous graphs (multiple node and edge types)
- Bipartite graphs (two disjoint node sets)
- Signed graphs (positive and negative edges)
- Hypergraphs (edges connecting multiple nodes)
- Dynamic graphs (graphs evolving over time)

Each of these settings requires specialized embedding approaches.

---

## UNIT 4: Graph Neural Networks (CORE)

### 12. What are Graph Neural Networks?
- GNNs learn node representations by iteratively aggregating
  information from neighboring nodes.
- They are trained end-to-end and optimized for specific tasks.
- GNNs overcome the limitations of shallow, task-agnostic embeddings.

---

### 13. General GNN Framework (Node-focused)
- GNNs follow the **message passing** paradigm.

Core steps:
- Message aggregation:  
  m_v^{(k)} = AGG({h_u^{(k−1)} | u ∈ N(v)})
- Update:  
  h_v^{(k)} = UPDATE(h_v^{(k−1)}, m_v^{(k)})

After K layers:
- Each node captures information from its K-hop neighborhood.

---

### 14. Graph-focused GNN Framework
- Used for graph-level tasks such as classification or regression.
- Two main steps:
  1. Learn node embeddings
  2. Aggregate node embeddings into a graph representation

Common readout functions:
- Sum
- Mean
- Max
- Attention-based pooling

---

### 15. Graph Filters

**Spectral-based filters**
- Defined in the frequency domain
- Based on Laplacian eigen-decomposition
- Theoretically strong but computationally expensive

**Spatial-based filters**
- Operate directly on node neighborhoods
- More scalable and widely used in practice

Examples:
- GCN
- GraphSAGE
- GAT

---

### 16. Graph Pooling

**Flat pooling**
- Global sum, mean, or max pooling
- Simple but loses structural information

**Hierarchical pooling**
- Learns graph coarsening in multiple stages
- Preserves hierarchical structure
- Examples: DiffPool, Top-K pooling

---

### 17. Parameter Learning in GNNs

**Node classification**
- Loss computed on labeled nodes
- Semi-supervised learning is common

**Graph classification**
- Loss computed on graph representations
- Pooling strategy plays a crucial role

---

## Key Takeaways
- Graph embedding methods are mostly task-agnostic
- GNNs are task-driven and end-to-end
- Spectral theory explains smoothing behavior in GNNs
- Spatial GNNs are dominant in industry applications

---


---

## UNIT 5: Robust Graph Neural Networks

While studying Graph Neural Networks further, I learned that GNNs can be
highly sensitive to noise or manipulation in the graph. Since GNNs rely
directly on graph structure for message passing, even small changes in
edges or node features can lead to incorrect predictions. This unit
summarizes how such attacks work and how GNNs can be made more robust.

---

### 18. Why Robustness Matters in GNNs
- GNNs propagate information through edges, so errors can spread quickly.
- A small change in the graph structure can influence many nodes.
- Real-world graphs are rarely clean and trustworthy.

Common real-world issues:
- Fake friendships in social networks
- Artificial clicks or interactions in recommender systems
- Malicious behavior in fraud detection graphs

Because of this, GNNs must be designed to remain stable under noisy or
adversarial conditions.

---

### 19. Graph Adversarial Attacks
- Graph adversarial attacks intentionally modify the graph to fool a GNN.
- The goal is usually to change predictions or reduce model accuracy.

Typical attack targets:
- Node-level predictions (e.g., misclassifying one user)
- Graph-level predictions (e.g., wrong molecule property)
- Overall model performance

Common attack methods:
- Adding fake edges
- Removing important edges
- Modifying node features

---

### 20. Types of Graph Adversarial Attacks

Graph attacks can be understood based on **when**, **what**, and **how**
the attacker acts.

Based on attack timing:
- Training-time attacks (poisoning the data before training)
- Test-time attacks (changing the graph after the model is trained)

Based on attack goal:
- Targeted attacks (specific node or graph)
- Untargeted attacks (general performance degradation)

Based on attacker knowledge:
- White-box attacks
- Gray-box attacks
- Black-box attacks

---

### 21. White-box Attacks
- The attacker has full knowledge of the GNN:
  - Architecture
  - Parameters
  - Gradients
- Attacks are often gradient-based and carefully optimized.

Typical behavior:
- Connecting a normal node to a malicious group
- Removing helpful neighbors

Key insight:
- Due to message passing, even one carefully chosen edge change
  can cause large prediction errors.

Limitation:
- Assumes unrealistic access to the model internals.

---

### 22. Gray-box Attacks
- The attacker has partial information about the model.
- Architecture or training data may be known, but parameters are hidden.

How the attack works:
1. Train a similar (surrogate) GNN.
2. Generate adversarial changes on the surrogate.
3. Apply those changes to the real model.

This works because graph attacks often transfer across models.

---

### 23. Black-box Attacks
- The attacker has no access to the model internals.
- Only observes model outputs such as labels or confidence scores.

Common strategies:
- Querying the model repeatedly
- Trial-and-error edge manipulation
- Heuristic or reinforcement-based approaches

Characteristics:
- Most realistic in practice
- Less powerful than white-box attacks
- Often more expensive in terms of queries

---

### 24. Defending Against Graph Attacks
- Defenses aim to reduce the impact of adversarial changes.
- Protection can be applied at different stages:
  - During training
  - Before feeding the graph to the model
  - Inside the model architecture itself

Main defense strategies:
- Adversarial training
- Graph purification
- Attention-based methods
- Graph structure learning

---

### 25. Graph Adversarial Training
- The model is trained using both clean and attacked graphs.
- This helps the model learn how to handle worst-case scenarios.

Advantages:
- Improves robustness against known attacks

Trade-offs:
- Higher training cost
- Possible slight drop in accuracy on clean data

---

### 26. Graph Purification
- The idea is to clean the graph before using it.
- Suspicious or noisy edges are removed based on heuristics.

Common techniques:
- Removing edges between very dissimilar nodes
- Filtering low-confidence connections
- Applying graph smoothing

Pros:
- Model-independent
- Simple to apply

Cons:
- Risk of removing useful edges

---

### 27. Using Attention for Robustness
- Attention mechanisms assign different importance to neighbors.
- Noisy or malicious neighbors naturally receive lower weights.

Why this helps:
- Reduces the influence of fake edges
- Improves stability without explicitly detecting attacks

Limitation:
- Attention weights themselves can still be targeted by attackers

---

### 28. Graph Structure Learning
- Instead of fully trusting the given graph, the model learns the structure.
- Edges are treated as learnable rather than fixed.

Benefits:
- Reduces reliance on noisy input graphs
- Learns task-relevant connections automatically

Cost:
- More parameters
- Higher computational complexity

---

### 29. Key Takeaways from Robust GNNs
- GNNs are vulnerable because of message passing.
- Attacks can be small but highly effective.
- Robustness requires:
  - Better training strategies
  - Cleaning or reweighting edges
  - Learning which neighbors to trust

Core idea:
Robust GNNs focus on **deciding which connections are reliable** rather than
blindly trusting the input graph.

---


## References

1. Yao Ma, Jiliang Tang.  
   *Deep Learning on Graphs*. Cambridge University Press, 2021.

2. (Discussed within the above book)
   - Kipf & Welling, GCN (ICLR 2017)
   - Hamilton et al., GraphSAGE (NeurIPS 2017)
   - Veličković et al., GAT (ICLR 2018)
   - Dai et al., Adversarial Attacks on Graph Neural Networks
   - Zügner et al., Adversarial Attacks on Node Classification




---

## Disclaimer
These notes are written in my own words for learning and revision purposes.
They are inspired by the book *Deep Learning on Graphs* by Yao Ma and Jiliang Tang.
No copyrighted text or figures have been reproduced.
