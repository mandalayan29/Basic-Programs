# Retrieval Augmented Generation (RAG)

Retrieval Augmented Generation (RAG) is a technique that helps you work with large documents that are too big to fit into a single prompt. Instead of cramming everything into one massive prompt, RAG breaks documents into chunks and only includes the most relevant pieces when answering questions.

---

## The Problem with Large Documents

Imagine you have an 800-page financial document and want to ask Claude specific questions about it, like *"What risk factors does this company have?"* You need to get the relevant information from the document to Claude somehow, but there are limits to how much text you can include in a prompt.

![Claude with large documents](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 1: Claude with large documents -->)

---

## Option 1: Include Everything in the Prompt

The first approach is straightforward — extract all text from the document and stuff it into your prompt along with the user's question. Your prompt might look like this:

```text
Answer the user's question about the financial document.

<user_question>
{user_question}
</user_question>

<financial_document>
{financial_document}
</financial_document>
```

![Option 1: Include the contents of the document in a prompt](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 2: Option 1 prompt stuffing -->)

### Limitations of Option 1
* There's a hard limit on prompt length — your document might be too long.
* Claude becomes less effective with very long prompts.
* Larger prompts cost more to process.
* Larger prompts take longer to process.

---

## Option 2: Break Documents into Chunks

RAG takes a smarter approach. First, you break the document into smaller chunks during a preprocessing step. Then, when a user asks a question, you find the chunks most relevant to their question and only include those in your prompt.

![Option 2: Break the document up into many chunks](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 3: Document chunking diagram -->)

Here's how it works: if someone asks *"What risks does this company face?"*, you'd search through your chunks, find the "Risk Factors" section, and include just that relevant chunk in your prompt.

![Option 2: Dynamic prompt assembly from relevant chunk](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 4: Selective chunk retrieval into prompt -->)

### Benefits of RAG
* Claude can focus on only the most relevant content.
* Scales up to very large documents.
* Works with multiple documents.
* Smaller prompts cost less and run faster.

### Challenges with RAG
* Requires a preprocessing step to chunk documents.
* Need a search mechanism to find "relevant" chunks.
* Included chunks might not contain all the context Claude needs.
* Many ways to chunk text — which approach is best?

For example, you could split documents into equal-sized portions, or you could create chunks based on document structure like headers and sections. Each approach has trade-offs you'll need to evaluate for your specific use case.

---

## When to Use RAG

RAG involves many technical decisions and requires more work than simply including everything in a prompt. You'll need to analyze whether the benefits outweigh the complexity for your particular application. It's especially valuable when working with very large documents, multiple documents, or when you need to optimize for cost and performance.

The key insight is that RAG trades simplicity for scalability and efficiency. While it requires more upfront work to implement properly, it enables you to work with document collections that would be impossible to handle with simple prompt stuffing.

---

## Text Chunking Strategies

Text chunking is one of the most critical steps in building a RAG pipeline. How you break up your documents directly impacts the quality of your entire system. A poor chunking strategy can lead to irrelevant context being inserted into your prompts, causing your AI to give completely wrong answers.

![Chunk source text overview](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 5: Chunk source text overview -->)

Consider this example: you have a document with sections on medical research and software engineering. If you chunk poorly, a user asking *"How many bugs did engineers fix this year?"* might get information about medical research instead of software engineering, simply because the medical section happened to contain the word "bug" in a different context.

![Poor chunking example](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 6: Poor chunking example with ambiguity -->)

This is why choosing the right chunking strategy matters so much. Let's explore three main approaches.

![Chunking Strategies comparison](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 7: Chunking Strategies summary table -->)

### 1. Size-Based Chunking

Size-based chunking is the simplest approach — you divide your text into strings of equal length. If you have a 325-character document, you might split it into three chunks of roughly 108 characters each.

![Size-based chunking diagram](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 8: 325 character split into 3 chunks -->)

#### Downsides of Size-Based Chunking
* Each chunk has some cutoff text (words get cut off mid-sentence).
* Each chunk lacks context.
* Section headers might be separated from their content.

![Size-based downsides](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 9: Text cutoffs and context loss -->)

#### Workaround: Overlapping Chunks
To address these issues, you can add overlap between chunks. This means each chunk includes some characters from the neighboring chunks, providing better context and ensuring complete words and sentences.

![Chunk overlap workaround](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 10: Neighboring chunk overlap mechanism -->)

```python
def chunk_by_char(text, chunk_size=150, chunk_overlap=20):
    chunks = []
    start_idx = 0
    
    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))
        chunk_text = text[start_idx:end_idx]
        chunks.append(chunk_text)
        
        start_idx = (
            end_idx - chunk_overlap if end_idx < len(text) else len(text)
        )
    
    return chunks
```

---

### 2. Structure-Based Chunking

Structure-based chunking divides text based on the document's natural structure — headers, paragraphs, and sections. This works great when you have well-formatted documents like Markdown files.

![Structure-based chunking diagram](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 11: Splitting text along markdown headers -->)

For a Markdown document, you can split on header markers:

```python
import re

def chunk_by_section(document_text):
    pattern = r"\n## "
    return re.split(pattern, document_text)
```

This approach gives you the cleanest, most meaningful chunks because each one represents a complete section. However, it only works when you have guarantees about your document structure. Many real-world documents are plain text or PDFs without clear structural markers.

---

### 3. Semantic-Based Chunking

Semantic-based chunking is the most sophisticated approach. You divide text into sentences, then use natural language processing to determine how related consecutive sentences are. You build chunks from groups of related sentences.

This method is computationally expensive but produces the most relevant chunks. It requires understanding the meaning of individual sentences and is more complex to implement than the other strategies.

---

### 4. Sentence-Based Chunking

A practical middle ground is chunking by sentences. You split the text into individual sentences using regular expressions, then group them into chunks with optional overlap:

```python
import re

def chunk_by_sentence(text, max_sentences_per_chunk=5, overlap_sentences=1):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(sentences):
        end_idx = min(start_idx + max_sentences_per_chunk, len(sentences))
        current_chunk = sentences[start_idx:end_idx]
        chunks.append(" ".join(current_chunk))
        
        start_idx += max_sentences_per_chunk - overlap_sentences
        
        if start_idx < 0:
            start_idx = 0
    
    return chunks
```

---

### Choosing Your Strategy

| Strategy | When to Use | Trade-offs |
| :--- | :--- | :--- |
| **Structure-based** | Best results when you control document formatting (e.g., internal company reports). | Requires strict formatting consistency; fails on flat text/PDFs. |
| **Sentence-based** | Good middle ground for standard text documents. | Context boundaries are cleaner than character cuts, but still arbitrary. |
| **Size-based** | Most reliable fallback; works with any content type, including code. | Can sever related context; requires overlap padding. |

> **Takeaway:** Size-based chunking with overlap is often the go-to choice in production because it is simple, reliable, and works across any document type.

#### Downloads
* `001_chunking.ipynb`
* `report.md`

---

## Finding Relevant Chunks

After breaking a document into chunks, the next step in a RAG pipeline is finding which chunks are most relevant to a user's question. This is essentially a search problem — you need to look through all your text chunks and identify the ones that relate to what the user is asking about.

![Finding Relevant Chunks](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 12: Finding Relevant Chunks query matching -->)

### Semantic Search

The most common approach for finding relevant chunks is semantic search. Unlike keyword-based search that looks for exact word matches, semantic search uses text embeddings to understand the meaning and context of both the user's question and each text chunk.

![Semantic Search Overview](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 13: Semantic search concept -->)

---

## Text Embeddings

A text embedding is a numerical representation of the meaning contained in some text. Think of it as converting words and sentences into a format that computers can work with mathematically.

![Text Embeddings mapping diagram](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 14: Text Embeddings flow diagram -->)

### How the Embedding Process Works:
1. You feed text into an embedding model.
2. The model outputs a long list of numbers (the embedding vector).
3. Each number ranges from $-1$ to $+1$.
4. These numbers represent different qualities or features of the input text.

### Understanding the Numbers

Each number in an embedding is essentially a "score" for some quality of the input text. However, we don't know precisely what each number represents.

![Interpreting embedding vector scores](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 15: Abstract dimension scores representation -->)

While it is helpful to conceptualize that one number might represent "how happy the text is" or "how much the text talks about oceans," these are purely conceptual examples. The actual meaning of each dimension is learned by the model during training and isn't directly human-interpretable.

---

## VoyageAI for Embeddings

Recommended provider: **VoyageAI**.

1. Sign up for a VoyageAI account.
2. Generate an API key.
3. Add the key to your `.env` file:

```env
ANTHROPIC_API_KEY="MY_KEY"
VOYAGE_API_KEY="MY_KEY"
```

![Environment variables setup](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 16: .env configuration in VS Code -->)

### Embedding Generation Implementation

```bash
%pip install voyageai python-dotenv
```

```python
from dotenv import load_dotenv
import voyageai

load_dotenv()
client = voyageai.Client()

def generate_embedding(text, model="voyage-3-large", input_type="query"):
    result = client.embed([text], model=model, input_type=input_type)
    return result.embeddings[0]
```

Executing this code on a chunk generates a vector of floating-point values:

```python
with open("./report.md", "r") as f:
    text = f.read()

chunks = chunk_by_section(text)
generate_embedding(chunks[0])
```

![Vector output inspect](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 17: Embeddings output array in Jupyter Notebook -->)

---

## The Complete RAG Pipeline Step-by-Step

Let's walk through an intuitive 2-dimensional embedding example to see the math and mechanics in action.

### Step 1: Chunk Your Source Text
* **Section 1: Medical Research** — *"This year saw significant strides in our understanding of XDR-47, a 'bug' we have not seen before."*
* **Section 2: Software Engineering** — *"This division dedicated significant effort to studying various infection vectors in our distributed systems"*

### Step 2: Generate Embeddings
Imagine a 2D embedding model where:
* Dimension 1: Degree to which the text discusses medicine.
* Dimension 2: Degree to which the text discusses software engineering.

![Imaginary 2D Embedding Model](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 18: 2D vector coordinate mapping -->)

* Medical Research chunk: $[0.97, 0.34]$
* Software Engineering chunk: $[0.30, 0.97]$

#### Normalization
Vectors are scaled to a unit magnitude of $1.0$:
* Medical Research normalized: $[0.944, 0.331]$
* Software Engineering normalized: $[0.295, 0.955]$

![Unit circle vector plot](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 19: Unit Circle Vector Representation -->)

---

### Step 3: Store in Vector Database

We store these normalized vectors alongside their text payload in a vector database.

![Vector Database Storage](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 20: Storing embeddings into vector store -->)

---

### Step 4: Process User Query

A user submits: *"I'm curious about the company. In particular, what did the software engineering dept do this year?"*

Passing this query through the embedding model outputs $[0.1, 0.89]$, which normalizes to $[0.112, 0.993]$.

![Embedding User Query](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 21: Query embedding pipeline -->)

---

### Step 5: Find Similar Embeddings (Cosine Similarity)

The vector database compares the query vector against all chunk vectors using **Cosine Similarity**:

![Vector query comparison on unit circle](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 22: Angular distance between query and document vectors -->)

$$\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \, \|\mathbf{B}\|}$$

![Cosine similarity calculations](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 23: Cosine math step-by-step -->)

$$\cos(a) = \frac{(0.112, 0.993) \cdot (0.295, 0.955)}{\|(0.112, 0.993)\| \, \|(0.295, 0.955)\|} = 0.983$$

$$\cos(b) = \frac{(0.112, 0.993) \cdot (0.944, 0.331)}{\|(0.112, 0.993)\| \, \|(0.944, 0.331)\|} = 0.398$$

* **Cosine Similarity scale:** Ranges from $-1$ (exact opposite) to $1$ (identical direction); $0$ indicates orthogonality (unrelated).
* **Cosine Distance:** Often calculated as:
  $$\text{Cosine Distance} = 1 - \text{Cosine Similarity}$$
  *(Lower distance = higher similarity)*

---

### Step 6: Create the Final Prompt

We inject the retrieved Software Engineering chunk directly into the context window:

![Final Prompt Assembly](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 24: Prompt injection into Claude -->)

```text
Answer the user's question about the financial document.

<user_question>
How many bugs did engineers fix this year?
</user_question>

<report>
## Section 2: Software Engineering
This division dedicated significant effort to studying various infection vectors in our distributed systems
</report>
```

---

## Complete End-to-End Implementation

```python
from dotenv import load_dotenv
import voyageai

load_dotenv()
client = voyageai.Client()

# 1. Chunk document
with open("./report.md", "r") as f:
    text = f.read()

chunks = chunk_by_section(text)

# 2. Generate embeddings in batch
embeddings = client.embed(chunks, model="voyage-3-large", input_type="document").embeddings

# 3. Store in Vector Index
store = VectorIndex()
for embedding, chunk in zip(embeddings, chunks):
    store.add_vector(embedding, {"content": chunk})

# 4. Embed user query
query = "What did the software engineering dept do last year?"
user_embedding = client.embed([query], model="voyage-3-large", input_type="query").embeddings[0]

# 5. Search for top-k matches
results = store.search(user_embedding, k=2)
for doc, distance in results:
    print(distance, "\n", doc["content"][:200], "\n")
```

![Vector search code notebook execution](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 25: Jupyter notebook search code -->)

---

## Hybrid Search: Combining Semantic and Lexical (BM25)

Pure semantic search can struggle with precise alphanumeric keywords, unique identifiers (e.g., `INC-2023-Q4-011`), or acronyms.

![Semantic search limitation with specific IDs](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 26: Hybrid search high-level concept -->)

### The Solution: Parallel Hybrid Search

Execute semantic search and lexical search simultaneously, then combine the candidate rankings.

![Parallel Search Architecture](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 27: Semantic + Lexical merge architecture -->)

### How BM25 Works

1. **Tokenize Query:** Deconstruct question into individual tokens (e.g., `["a", "INC-2023-Q4-011"]`).
2. **Frequency Count:** Count term occurrences across corpus.
3. **Inverse Document Frequency (IDF) Weighting:** Frequent stop words receive very low weights; rare IDs receive high weights.
4. **Rank Matching Documents:** Rank chunks containing the highest aggregate score of weighted terms.

![BM25 Scoring Process Flow](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 28: BM25 step-by-step scoring breakdown -->)

```python
# Implementing BM25 Search
store = BM25Index()
for chunk in chunks:
    store.add_document({"content": chunk})

results = store.search("What happened with INC-2023-Q4-011?", 3)
for doc, distance in results:
    print(distance, "\n", doc["content"][:200], "\n----\n")
```

![BM25 execution results](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 29: BM25 exact match output in terminal -->)

---

## Merging Results with Reciprocal Rank Fusion (RRF)

Because Vector search produces cosine distances while BM25 produces unbounded matching scores, their raw scores cannot be directly averaged. **Reciprocal Rank Fusion (RRF)** normalizes results purely based on ranking positions.

![Rank list from VectorIndex and BM25Index](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 30: Vector vs BM25 individual ranks -->)

### RRF Formula

$$\text{RRF\_score}(d) = \sum_{i=1}^{n} \frac{1}{k + \text{rank}_i(d)}$$

Where:
* $k$ is a smoothing constant (standard default is $60$, set to $1$ below for illustration).
* $\text{rank}_i(d)$ is the 1-based index rank of document $d$ from search system $i$.

![RRF Calculation Table](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 31: RRF Calculation Breakdown -->)

### Calculation Example ($k = 1$):

| Text Chunk | Rank from Vector | Rank from BM25 | RRF Score Formula | Final Score | Final Rank |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **Section 2: Software Engineering...** | 1 | 2 | $\frac{1.0}{1 + 1} + \frac{1.0}{1 + 2} = 0.500 + 0.333$ | **0.833** | **1** |
| **Section 6: Product Engineering...** | 3 | 1 | $\frac{1.0}{1 + 3} + \frac{1.0}{1 + 1} = 0.250 + 0.500$ | **0.750** | **2** |
| **Section 7: Historical Research...** | 2 | 3 | $\frac{1.0}{1 + 2} + \frac{1.0}{1 + 3} = 0.333 + 0.250$ | **0.583** | **3** |

![Final Ranked Result Table](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 32: Final RRF rankings -->)

---

## Unified Hybrid Retriever Architecture

```python
from typing import Dict, Any, List

class Retriever:
    def __init__(self, *indexes):
        if len(indexes) == 0:
            raise ValueError("At least one index must be provided")
        self._indexes = list(indexes)
    
    def add_document(self, document: Dict[str, Any]):
        for index in self._indexes:
            index.add_document(document)
    
    def search(self, query_text: str, k: int = 3, k_rrf: int = 60):
        rrf_scores = {}
        doc_lookup = {}
        
        for index in self._indexes:
            results = index.search(query_text, k=k * 2)
            for rank, (doc, _) in enumerate(results, start=1):
                doc_id = doc.get("id") or doc["content"]
                doc_lookup[doc_id] = doc
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_rrf + rank))
        
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return [(doc_lookup[doc_id], score) for doc_id, score in sorted_docs[:k]]
```

![Modular Retriever Architecture](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=<!-- Replace with base64 string for Slide 33: Retriever wrapping arbitrary index providers -->)

### Extensibility
This architecture supports seamless plug-and-play expansion. You can plug in keyword search, vector stores, BM25 indices, or knowledge graphs by conforming to the same simple interface (`add_document` and `search`).