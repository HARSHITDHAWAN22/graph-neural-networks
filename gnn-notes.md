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

## References

1. Yao Ma, Jiliang Tang.  
   *Deep Learning on Graphs*. Cambridge University Press, 2021.

2. (Discussed within the above book)
   - Kipf & Welling, GCN (ICLR 2017)
   - Hamilton et al., GraphSAGE (NeurIPS 2017)
   - Veličković et al., GAT (ICLR 2018)




---

## Disclaimer
These notes are written in my own words for learning and revision purposes.
They are inspired by the book *Deep Learning on Graphs* by Yao Ma and Jiliang Tang.
No copyrighted text or figures have been reproduced.
