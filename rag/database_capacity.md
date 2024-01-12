### Bounding the Vector Database

In the proof of concept (POC) phase, establishing meaningful lower and upper
bounds for the size/capacity of the vector database requires certain
foundational assumptions.

Key among these is the premise that irrespective of the raw data format - be it
images, text, or tabular data - we can uniformly transform them into a vector
representation. Specifically, this transformation process will yield vectors
$\mathbf{x} \in \mathbb{R}^{D}$, where $D$ represents the dimensionality.

Here are some notations to facilitate the discussion, and I will assume a static
ingestion due to my lackthereof understanding on infrastructure (yes I can do my
best in estimating but I will not know what I do not know):

1. $N$: The number of vectors in the database.

2. $D$: The dimensionality of each vector.

3. $\mathcal{X}$: The set of all vectors in the database,
   $\mathcal{X} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\}$.

4. $\mathbf{x}_n$: The $n$-th vector in the database, where
   $\mathbf{x}_n \in \mathcal{X}$.

5. $\mathcal{E}$: The set of embedding models used to generate the vectors.

6. $\mathcal{M}_n$: The metadata associated with vector $\mathbf{x}_n$. This can
   include a variety of information such as source, creation date, labels, etc.

7. $ID_{\mathbf{x}_n}$: The unique identifier associated with vector
   $\mathbf{x}_n$.

8. $S_{\text{ID}}$: The size (in bytes) of each ID.

9. $\mathcal{I}$: The index structure used to manage and query the vectors in
   the database.

10. $S_{\mathcal{I}}$: The total size (in bytes) of the index structure
    $\mathcal{I}$.

11. $S_{\text{data type}}$: The size (in bytes) of the data type used for each
    element in the vectors.

12. $S_{\mathcal{M}_n}$: The average size (in bytes) of the metadata associated
    with each vector.

$$
\text{Total Storage} = (N \times D \times S_{\text{data type}}) + (N \times S_{\text{ID}}) + (N \times S_{\mathcal{M}_n}) + S_{\mathcal{I}}
$$

---

| Notation               | Description                                                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| $N$                    | The number of vectors in the database.                                                                                    |
| $D$                    | The dimensionality of each vector.                                                                                        |
| $\mathcal{X}$          | The set of all vectors in the database, $\mathcal{X} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\}$.             |
| $\mathbf{x}_n$         | The $n$-th vector in the database, where $\mathbf{x}_n \in \mathcal{X}$.                                                  |
| $\mathcal{E}$          | The set of embedding models used to generate the vectors.                                                                 |
| $\mathcal{M}_n$        | The metadata associated with vector $\mathbf{x}_n$. This can include information like source, creation date, labels, etc. |
| $ID_{\mathbf{x}_n}$    | The unique identifier associated with vector $\mathbf{x}_n$.                                                              |
| $S_{\text{ID}}$        | The size (in bytes) of each ID.                                                                                           |
| $\mathcal{I}$          | The index structure used to manage and query the vectors in the database.                                                 |
| $S_{\mathcal{I}}$      | The total size (in bytes) of the index structure $\mathcal{I}$.                                                           |
| $S_{\text{data type}}$ | The size (in bytes) of the data type used for each element in the vectors.                                                |
| $S_{\mathcal{M}_n}$    | The average size (in bytes) of the metadata associated with each vector.                                                  |

How do we derive $N$?

We also have some basic assumptions:

-   Number of reports: 100 but can be more
-   Each annual report have on average 250 pages
-   Each page have on average 100 sentences with 15 words per sentence.
-   Each report have on average 20 images.

---

Yes, metadata, IDs, and indexes do take up space and memory in a vector
database, and their impact on the overall storage requirements can be
significant, especially at scale. Let's break down these components:

1. **Metadata**:

    - Metadata refers to additional information about each vector, such as its
      source, creation date, or associated labels.
    - The size of metadata varies depending on its complexity and the amount of
      information stored. For instance, a few additional bytes might be needed
      for basic information like timestamps, whereas more complex metadata
      (e.g., associated text descriptions or tags) could require substantially
      more space.

2. **IDs**:

    - Each vector in the database is typically associated with a unique
      identifier (ID) to facilitate efficient retrieval and management.
    - The memory requirement for IDs depends on the data type used for them. For
      example, a 32-bit integer ID requires 4 bytes, whereas a 64-bit integer
      requires 8 bytes. If UUIDs (Universally Unique Identifiers) are used, the
      requirement is even higher, typically 16 bytes per UUID.

3. **Indexes**:
    - Indexes are critical for efficient querying and retrieval of vectors from
      the database.
    - The size of an index depends on the indexing method and the number of
      vectors. Some indexing strategies create complex structures to speed up
      search operations, which can consume a significant amount of memory.
    - For large-scale databases, indexing can consume as much, if not more,
      memory as the data itself, particularly for methods optimized for
      high-dimensional and nearest-neighbor searches.

When estimating the capacity requirements for a vector database, it's important
to account for these additional components. The total storage requirement is not
just a function of the number of vectors $N$ and their dimensionality $D$, but
also includes the cumulative size of metadata, IDs, and indexes.

A simplified way to estimate the total storage requirement would be:

$$
\text{Total Storage} = (N \times D \times \text{Size of Data Type}) + (N \times \text{Size of ID}) + \text{Size of Metadata} + \text{Size of Indexes}
$$

Keep in mind that this is a simplified estimation. The actual storage
requirement could vary based on the database design, the efficiency of the
indexing algorithms, and the specific types of metadata used. Additionally,
database management systems may introduce their own overheads.

## Indexing

Indexing in a vector database plays a crucial role in efficiently managing and
retrieving high-dimensional data, like the vectors generated by embedding
models. Let's go through a concrete example to illustrate how indexing works in
a vector database:

### Example Scenario: Image Retrieval in a Vector Database

**Context**: Imagine you have a vector database that stores embeddings of
images. Each image is converted into a high-dimensional vector (say 512
dimensions) using an embedding model. You have a large number of these image
vectors stored in your database.

**Goal**: Your goal is to quickly find images that are visually similar to a
given query image.

**Process Without Indexing**:

-   Without indexing, to find the most similar images to a query image, you
    would have to compare the query vector with every single vector in the
    database.
-   This means performing millions (or more) vector comparisons, which is
    computationally expensive and time-consuming.

**Implementing Indexing**:

-   You decide to use an indexing method, such as HNSW (Hierarchical Navigable
    Small World), to organize these vectors.

**How Indexing Works**:

1. **Building the Index**: During the index-building phase, the HNSW algorithm
   organizes the vectors into a multi-layered graph structure. Each vector is a
   node in this graph, and nodes are connected to their 'neighbors' in a way
   that reflects their similarity.

2. **Efficient Search**: When you query the database with a new image vector,
   the index uses this graph structure to navigate efficiently through the
   vectors. It quickly traverses the layers and paths within the graph to find
   the nodes (vectors) most similar to your query.

3. **Retrieving Results**: The index significantly reduces the number of
   comparisons needed to find the nearest neighbors. Instead of comparing the
   query vector with all vectors in the database, it only compares with a small
   subset, leading to much faster retrieval times.

### Benefits of Indexing:

-   **Speed**: Dramatically faster queries for similar images.
-   **Scalability**: Enables the database to handle large volumes of
    high-dimensional data efficiently.
-   **Resource Efficiency**: Reduces the computational and memory resources
    required for queries.

In this example, the indexing structure transforms a potentially unmanageable
brute-force search problem into a feasible and efficient task, enabling
practical and fast image retrieval based on visual similarity.

## Why cannot we use Vectorization?

Using PyTorch's vectorization capabilities to find a vector similar to a query
vector is certainly possible and can be efficient, especially for smaller
datasets or lower-dimensional data. However, the situation becomes more complex
and challenging with large-scale, high-dimensional data, which is common in
applications involving vector databases. Here's why specialized indexing
techniques are often preferred in such scenarios:

### PyTorch Vectorization for Similarity Search:

-   **Efficiency**: PyTorch excels at performing vectorized operations, which
    are highly optimized and can be parallelized, especially on GPUs. This makes
    it suitable for batch operations on vectors, like calculating distances or
    similarities.
-   **Limitation with Scale**: As the number of vectors (N) and their
    dimensionality (D) grow, the computational load increases significantly. For
    a database with millions of high-dimensional vectors, a brute-force search
    (comparing a query vector with every vector in the database) becomes
    prohibitively expensive in terms of computation time and resources.
-   **Memory Constraints**: Storing all vectors in memory for vectorized
    operations might not be feasible for very large datasets, leading to
    additional challenges in data management and processing.

### Why Specialized Indexing is Often Used:

1. **Efficient Large-Scale Search**:

    - Indexing structures like KD-trees, HNSW, or LSH are specifically designed
      to handle large-scale, high-dimensional data. They provide mechanisms to
      efficiently narrow down the search space, reducing the number of required
      comparisons.

2. **Handling High-Dimensionality**:

    - Many indexing algorithms offer strategies to cope with the "curse of
      dimensionality," where the efficiency of traditional search algorithms
      deteriorates as the dimensionality increases.

3. **Query Speed**:

    - For applications requiring near real-time responses, such as recommender
      systems or image retrieval systems, indexing structures can provide
      responses much faster than a full scan of the database.

4. **Scalability and Flexibility**:
    - Indexing methods can often handle dynamic data where new vectors are added
      or existing ones are modified. They can scale to accommodate growing
      datasets without a significant drop in query performance.

### Conclusion:

-   **Small to Medium-Sized Data**: For smaller datasets or when high precision
    is required and computational resources are sufficient, PyTorch's
    vectorization can be an effective method for similarity search.
-   **Large-Scale High-Dimensional Data**: For very large datasets or
    high-dimensional vectors, specialized indexing techniques are often
    necessary to ensure efficient and scalable performance.

In summary, while PyTorch vectorization is powerful for certain applications,
the demands of large-scale vector databases often necessitate more sophisticated
indexing approaches to ensure efficiency and scalability.

## Embedding Model Output Dimension

Your understanding of the embedding model $\mathcal{E}$ as a function mapping a
series of tokens to a vector space is generally correct, but there seems to be a
bit of confusion regarding the output dimensions, especially in the context of
typical Retrieval-Augmented Generation (RAG) and large language model (LLM)
problems. Let's clarify:

### Embedding Model in RAG and LLM Context:

1. **Input Dimensionality**:

    - The input to the embedding model is a series of tokens. For example, if
      you input a sentence, it's first tokenized. Let's say the sentence has $L$
      tokens.

2. **Output Dimensionality**:
    - A common misconception is expecting an output dimension of $L \times D$.
      However, in many embedding models, especially those used in RAG and LLMs,
      the model outputs a single vector representing the entire input sequence
      (in your case, the entire chunk).
    - This output vector is of size $1 \times D$, where $D$ is the
      dimensionality of the embedding space.
    - The model effectively condenses the information from the $L$ tokens into a
      single $D$-dimensional vector.

### Visualization of Chunk as a Single Vector:

-   Imagine your chunk as a sentence with $L$ words.
-   After passing this chunk through the embedding model, you don't get $L$
    separate vectors. Instead, the model provides a single vector that
    encapsulates the meaning of the entire chunk.
-   This single vector is what gets stored in your vector database.

### Example for Clarification:

-   Let's say you have a chunk consisting of 10 words (i.e., $L = 10$).
-   Your embedding model $\mathcal{E}$ is designed to process chunks and output
    a single vector.
-   After processing this 10-word chunk, the model outputs a vector of size
    $1 \times D$ (not $10 \times D$).

In the context of RAG and LLMs, such embedding models are common because they
are designed to capture the contextual and semantic information of entire
sequences (like sentences or paragraphs) rather than individual tokens. This
approach is more aligned with the way language is naturally structured and
understood.
