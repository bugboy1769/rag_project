# Nova: Agentic GraphRAG System

A modular RAG system that bridges the gap between **Semantic Search** (content) and **Topological Reasoning** (structure).

## Current Architecture: Semantic Graph RAG

We have moved beyond simple text-chunking to a **Dual-Store Architecture**:

1.  **Vector Store (`vectordb.pkl`)**:
    *   **Content**: Raw text chunks (Paragraphs).
    *   **Purpose**: Retrives dense context and specific details.
    *   **Model**: `bge-base-en-v1.5`

2.  **Triplet Store (`tripletdb.pkl`)**:
    *   **Content**: LLM-extracted knowledge triplets (e.g., "Cat has_color Orange").
    *   **Purpose**: **Proposition Retrieval**. Matches specific facts and relationships semantically, filtering out noise from high-degree nodes (e.g., "Cat") that purely topological search would retrieve.
    *   **Mechanism**: Sentenized triplets encoded into the same vector space.

3.  **Graph Store (`graph.pkl`)**:
    *   **Content**: NetworkX DiGraph.
    *   **Purpose**: Holds raw topology (nodes/edges) for future traversal and visualization.

## Roadmap: Graph Representation Learning

We are addressing the limitation where "disconnected" facts (2-hop neighbors) are semantically distant.

### Phase A: Node2Vec (Topological Embeddings)
*   **Goal**: Learn embeddings based purely on *random walks* (structure), ignoring text content.
*   **Hypothesis**: Nodes that share structural roles or communities will be clustered together, enabling "structural retrieval".

### Phase B: Graph Convolutional Networks (GCNs)
*   **Goal**: Fuse **Content** + **Structure**.
*   **Mechanism**: Message passing layers where a node aggregates features from its neighbors.
*   **Enriched Embeddings**: The final vector for "Cat" will mathematically contain information propagated from "Heart" -> "Mammal" -> "Cat", enabling implicit multi-hop reasoning.

### Comparison Study
We will implement both modules to benchmark:
1.  **Naive RAG** (Baseline)
2.  **Semantic Graph RAG** (Current)
3.  **Node2Vec** (Structure Only)
4.  **GCN** (Structure + Content)
