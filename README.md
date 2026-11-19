# 🧠 LLM Pipeline Explorer

An interactive, visual, single-file educational laboratory designed to demystify how Large Language Models (LLMs) turn raw human text into numbers, geometry, vectors, and semantic meaning.

---

## 🌟 Quick Start

Since the entire application is completely self-contained in a single file:

1. Double-click or open **`index.html`** in any modern web browser (Chrome, Firefox, Safari, Edge).
2. No Node.js, Python, server installation, or build tools are required!

---

## 🚀 Core Features

### 1. 📖 Vocabulary & Tokenizer Explorer
* **Byte-Pair Encoding (BPE) & Subword Segmentation**: Type any sentence or prompt and see how it decomposes into subwords, prefix tokens, and integer `input_ids`.
* **Vocabulary Lookup Dictionary**: Search and filter model tokens by ID, string, semantic cluster, and vector norm ($||v||$).
* **Special & Subword Tokens**: Understand how models represent spaces, prefixes, suffixes (e.g. `##`), and maintain a bounded vocabulary (~32k–128k tokens) without suffering out-of-vocabulary (OOV) crashes.

### 2. 🧬 Continuous Embeddings & Vector Arithmetic
* **16-Dimensional Semantic Latent Space**: Inspect the continuous vector representation across 16 interpretable semantic latent dimensions:
  * *Royalty, Technology, Animacy, Fauna, Gender, Scale/Size, Abstractness, Nutrition, Geography, Sentiment, Dynamism, Modernity, Ecology, Cognition*.
* **Embedding Matrix Lookup**: Visualize the tensor lookup operation $E[\text{token\_id}] \in \mathbb{R}^{d}$.
* **Vector Algebra Lab ("King - Man + Woman = Queen")**:
  $$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$
  Perform live vector additions and subtractions and see the top nearest-neighbor vocabulary matches ranked by cosine similarity.

### 3. 🌌 Interactive 3D Semantic Vector Space
* **WebGL-Accelerated 3D Scatter Plot**: Rotate, zoom, pan, and hover over semantic word clusters in 3D space.
* **On-the-Fly Dimensionality Reduction (PCA)**:
  * Computes the 16×16 covariance matrix in JavaScript.
  * Solves for the top 3 principal eigenvectors via Power Iteration with Gram-Schmidt orthogonal deflation.
  * Dynamically projects high-dimensional embeddings down to 3D $(x, y, z)$ coordinates.
* **Alternative Projections**:
  * **PCA 3D**: Captures maximum mathematical variance across all dimensions.
  * **Semantic Axis 3D**: Direct alignment (X: Royalty vs Food, Y: Tech vs Animal, Z: Nature vs Emotion).
  * **Unit Sphere Projection**: Visualizes normalized cosine angular geometry.
* **Distance Inspector & Vector Line**:
  * Select any two points to draw a direct 3D vector connector.
  * Live readout of Cosine Similarity $\cos(\theta)$, angular separation in degrees ($\theta^\circ$), and Euclidean distance ($L_2$).

### 4. ➕ Adding Your Own Custom Words
* **Custom Token Injection**: Add novel words, concepts, or fictional entities to the model's vocabulary.
* **Two Creation Modes**:
  1. **Manual Semantic Sliders**: Fine-tune each of the 16 latent dimensions from $-1.00$ to $+1.00$.
  2. **Semantic Blend (Linear Interpolation)**: Synthesize a vector by blending two existing words with a slider:
     $$\mathbf{v}_{\text{new}} = (1 - \alpha)\mathbf{v}_{\text{word}_1} + \alpha \mathbf{v}_{\text{word}_2}$$
     *(e.g., blend "robot" and "man" to synthesize "cyborg")*.
* **Instant Dynamic Update**: The new word is immediately added to the vocabulary table, projected into 3D space via PCA, and made available for vector math and distance comparisons.

### 5. 🔍 Cosine Similarity Matrix & Distance Lab
* **Pairwise Similarity Heatmaps**: Interactive heatmaps comparing semantic clusters (Royalty vs Animals, Tech vs Food, or full dense matrix).
* **Top-K Nearest Neighbors (k-NN)**: Rapidly identify the closest semantic neighbors for any word in the vocabulary.

### 6. 🎓 End-to-End Pipeline Walkthrough
Step-by-step interactive architectural guide explaining the 5 fundamental stages of Transformer inference:
1. **Raw Text $\to$ Tokenization**
2. **Token IDs $\to$ Embedding Matrix Lookup**
3. **Positional Encoding Addition ($E_{\text{tok}} + E_{\text{pos}}$)**
4. **Multi-Head Self-Attention ($\text{softmax}(QK^T / \sqrt{d_k})V$)**
5. **Output Projection, Unembedding, Logits & Softmax Sampling**

---

## 📐 Mathematical Foundations

### 1. Cosine Similarity vs. Euclidean Distance
In high-dimensional spaces, vector magnitude is often less informative than directional alignment. Cosine similarity evaluates the angle $\theta$ between vectors $\mathbf{u}$ and $\mathbf{v}$:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}} = \cos(\theta)$$

$$\text{Angular Separation} = \theta = \arccos\left(\text{Cosine Similarity}\right)$$

$$\text{Euclidean Distance} = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{d} (u_i - v_i)^2}$$

### 2. Transformer Self-Attention Dot Product
The scaled dot-product attention at the heart of Large Language Models is an unnormalized cosine similarity scaled by the square root of key dimension $d_k$:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

### 3. Principal Component Analysis (PCA)
To project 16 dimensions down to 3 dimensions for display, the app computes:
1. Centered data: $\mathbf{X}_c = \mathbf{X} - \boldsymbol{\mu}$
2. Sample Covariance: $\mathbf{\Sigma} = \frac{1}{n - 1}\mathbf{X}_c^T \mathbf{X}_c$
3. Top 3 eigenvectors $\mathbf{w}_1, \mathbf{w}_2, \mathbf{w}_3$ via power iteration with deflation:
   $$\mathbf{y}_i = \left( \mathbf{x}_i \cdot \mathbf{w}_1, \; \mathbf{x}_i \cdot \mathbf{w}_2, \; \mathbf{x}_i \cdot \mathbf{w}_3 \right)$$

---

## 📁 Repository Structure

```
LLM Pipeline Explorer/
├── index.html   # Fully self-contained single-page application (UI, 3D Canvas, PCA engine)
└── README.md    # Comprehensive educational guide & technical documentation
```

*Built as an educational laboratory to make the internal mechanics of LLMs tangible and intuitive.*
