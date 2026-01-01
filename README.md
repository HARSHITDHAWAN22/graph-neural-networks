# Graph Neural Networks — My Learning Notes

This repository contains my complete learning notes on **Graph Neural Networks (GNNs)**.
I created these notes while studying GNNs step by step, starting from basic ideas
and gradually moving towards advanced topics and real-world applications.

The purpose of this repository is simple:
- To clearly understand how GNNs work
- To build strong intuition before jumping into heavy coding
- To keep everything in one place for revision and future reference

My primary learning source for these notes has been the book  
**Deep Learning on Graphs** by Yao Ma and Jiliang Tang.

---

## Why Graph Neural Networks?

Many real-world problems are not independent in nature. Instead, they involve
entities that are connected to each other.

Some common examples:
- Users interacting with products
- Accounts sending money to each other
- Molecules made of atoms and bonds
- Papers connected through citations

Traditional machine learning models usually ignore these relationships.
Graph Neural Networks are designed specifically to **use both data and connections**
to learn better representations.

---

## What I Have Covered in This Repository

### Foundations
- What graphs are and how they are represented
- Why traditional ML struggles with graph data
- Basic graph concepts like nodes, edges, adjacency, and neighborhoods
- How GNNs extend ideas from CNNs and neural networks to graphs

---

### Core Graph Neural Networks
- General GNN message passing framework
- Difference between node-level and graph-level tasks
- Popular GNN models:
  - GCN
  - GraphSAGE
  - GAT
- Spectral vs spatial graph convolutions
- Graph pooling techniques (global and hierarchical)

---

### Graph Embedding
- Why graph embeddings were used before GNNs
- Node co-occurrence based methods
- Structural role and community-based embeddings
- Limitations of shallow embeddings compared to GNNs

---

### Robust Graph Neural Networks
- Why GNNs are sensitive to small graph changes
- Graph adversarial attacks:
  - White-box
  - Gray-box
  - Black-box
- Defense techniques:
  - Adversarial training
  - Graph purification
  - Attention-based robustness
  - Learning graph structure instead of trusting it blindly

---

### Scalable Graph Neural Networks
- Why vanilla GNNs do not scale to large graphs
- Neighborhood explosion problem
- Sampling-based solutions:
  - Node-wise sampling (GraphSAGE)
  - Layer-wise sampling (FastGCN)
  - Subgraph-wise sampling (GraphSAINT)
- Trade-offs between accuracy and efficiency
- Why scalability is critical for industry systems

---

### Graph Neural Networks on Complex Graphs
- Heterogeneous graphs with multiple node and edge types
- Bipartite graphs used in recommender systems
- Multi-dimensional graphs with multiple relations
- Signed graphs with positive and negative edges
- Hypergraphs for group-level interactions
- Dynamic and temporal graphs that evolve over time

---

### Advanced Topics in GNNs
- Why deep GNNs are hard to train
- Over-smoothing problem
- Techniques to build deeper GNNs:
  - Jumping Knowledge
  - DropEdge
  - PairNorm
- Using unlabeled graph data through self-supervised learning
- Expressiveness limits of GNNs
- Weisfeiler–Lehman (WL) test and what GNNs can or cannot distinguish

---

### Advanced Applications of GNNs
- Using GNNs for combinatorial optimization problems
- Learning representations of computer programs
- Modeling interacting physical systems
- Viewing GNNs as neural reasoning models rather than just predictors

---

## Who This Repository Is For

This repository is useful for:
- Students trying to understand GNNs conceptually
- ML / AI engineers entering graph-based learning
- Anyone who wants strong intuition before implementation
- Revision and interview preparation

The focus here is **understanding first, coding later**.

---

## Key Learnings

- GNNs explicitly model relationships, not just features
- Graph structure is as important as node features
- Scalability and robustness are major real-world challenges
- Not all neighbors should be treated equally
- GNNs can be used for reasoning, not just prediction

---

## References

### Primary Source
1. Yao Ma, Jiliang Tang  
   *Deep Learning on Graphs*. Cambridge University Press, 2021.

### Discussed within the above book
- Kipf & Welling, GCN (ICLR 2017)
- Hamilton et al., GraphSAGE (NeurIPS 2017)
- Veličković et al., GAT (ICLR 2018)
- Chen et al., FastGCN
- Zeng et al., GraphSAINT
- Xu et al., How Powerful are Graph Neural Networks?
- Morris et al., Weisfeiler and Leman Go Neural
- Rong et al., DropEdge
- Xu et al., Jumping Knowledge Networks

These references are included as a learning roadmap and are discussed or
surveyed within the primary book.

---

## Disclaimer

These notes are written in my own words for learning and revision purposes.
They are based on my understanding of the book *Deep Learning on Graphs*
and related concepts. No copyrighted text or figures have been copied.

---

## Next Steps

Possible future additions to this repository:
- Hands-on GNN projects
- Implementations using PyTorch Geometric
- Experiments on real-world datasets
- Performance comparisons and case studies

---
