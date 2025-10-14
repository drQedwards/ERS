import json
import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# -----------------------------------------------------------------------------
# Parameter definitions
LAMBDA_BASE = 0.001
ALPHA = 0.1
BETA = 0.5
GAMMA = 0.2

SIM_THRESHOLD = 0.5
CONTRADICT_THRESHOLD = 0.5

HIDDEN_DIM = 256   # Dimensionality of compressed embeddings
NUM_PETALS = 8     # Number of petals in the attention flower


@dataclass
class MemoryBlock:
    """Container for a single memory entry."""
    content: str
    source_quality: float = 0.8
    volatility: float = 0.1
    confidence: float = 1.0
    timestamp: float = field(default_factory=lambda: time.time())
    access_count: int = 0
    embedding: Optional[np.ndarray] = None
    status: str = 'ACTIVE'

    def to_dict(self) -> Dict[str, any]:
        d = asdict(self)
        if self.embedding is not None:
            d['embedding'] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> 'MemoryBlock':
        emb = data.get('embedding')
        if emb is not None:
            data['embedding'] = np.array(emb)
        return cls(**data)


class PMLLLattice:
    """
    Approximates the Persistent Memory Logic Lattice using TF‑IDF, SVD and
    random multi‑petal transformations.

    * The vectorizer converts text into a sparse term frequency–inverse
      document frequency matrix.
    * The SVD reduces this matrix to `hidden_dim` dimensions (low‑rank
      compression).
    * The attention flower applies `num_petals` randomly initialised linear
      transformations (petals) to the compressed vectors, applies ReLU, and
      averages the results.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_petals: int = NUM_PETALS) -> None:
        # Desired hidden dimension. Actual dimension may be smaller if the TF‑IDF feature
        # space is small. We store both the desired dimension and the current
        # dimension separately.
        self.desired_hidden_dim = hidden_dim
        # Current hidden dimension (updated after each fit)
        self.hidden_dim = hidden_dim
        self.num_petals = num_petals
        self.vectorizer = TfidfVectorizer(stop_words='english')
        # Initialise SVD with dummy component count; will be replaced on fit.
        self.svd = TruncatedSVD(n_components=1, algorithm='randomized', random_state=42)
        # Petal weights (initialised during first fit)
        self.petals: Optional[np.ndarray] = None  # shape (num_petals, hidden_dim, hidden_dim)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer and SVD on texts and return processed embeddings."""
        # Step 1: TF‑IDF
        tfidf = self.vectorizer.fit_transform(texts)
        n_samples, n_features = tfidf.shape
        # Determine number of SVD components: cannot exceed min(n_samples, n_features) - 1
        # and should not exceed desired hidden dimension. Ensure at least 1 component.
        max_components = max(1, min(n_samples, n_features) - 1)
        n_components = min(self.desired_hidden_dim, max_components)
        # Update hidden_dim and SVD accordingly
        self.hidden_dim = n_components
        self.svd = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=42)
        # Step 2: SVD compression
        compressed = self.svd.fit_transform(tfidf)
        # Step 3: initialise petal weights if needed, or adjust if dimension changed
        if self.petals is None or self.petals.shape[1] != n_components:
            # (num_petals, hidden_dim, hidden_dim)
            self.petals = np.random.randn(self.num_petals, n_components, n_components) * 0.1
        # Step 4: multi‑petal attention
        processed = self._apply_petals(compressed)
        # L2 normalisation
        processed = normalize(processed)
        return processed

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform new texts into processed embeddings without refitting."""
        tfidf = self.vectorizer.transform(texts)
        compressed = self.svd.transform(tfidf)
        processed = self._apply_petals(compressed)
        processed = normalize(processed)
        return processed

    def _apply_petals(self, compressed: np.ndarray) -> np.ndarray:
        """Apply multi‑petal transformations to compressed vectors."""
        # compressed: (n_samples, hidden_dim)
        # petals: (num_petals, hidden_dim, hidden_dim)
        if self.petals is None:
            raise ValueError("Petal weights have not been initialised. Call fit_transform first.")
        # Expand compressed to (num_petals, n_samples, hidden_dim)
        # For each petal, compute linear transformation
        # We will accumulate results across petals
        n_samples = compressed.shape[0]
        results = np.zeros((n_samples, self.hidden_dim), dtype=float)
        for p in range(self.num_petals):
            W = self.petals[p]
            # Linear transform
            out = compressed @ W.T  # shape (n_samples, hidden_dim)
            # ReLU
            out = np.maximum(out, 0.0)
            # accumulate
            results += out
        # average across petals
        results /= self.num_petals
        return results


class MemoryStore:
    """Stores memory blocks and their embeddings."""

    def __init__(self) -> None:
        self.memories: List[MemoryBlock] = []

    def add(self, mem: MemoryBlock) -> None:
        self.memories.append(mem)

    def __len__(self) -> int:
        return len(self.memories)

    def find_similar(self, index: int, threshold: float = SIM_THRESHOLD) -> List[Tuple[int, float]]:
        """Find indices of memories similar to memory at `index`.
        Returns a list of (other_index, similarity).
        """
        target_mem = self.memories[index]
        if target_mem.embedding is None:
            return []
        target = target_mem.embedding
        # Precompute dot product with all memory embeddings
        sims: List[Tuple[int, float]] = []
        for i, mem in enumerate(self.memories):
            if i == index or mem.embedding is None:
                continue
            sim = float(np.dot(target, mem.embedding))
            if sim > threshold:
                sims.append((i, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims

    def to_dict(self) -> List[Dict[str, any]]:
        return [m.to_dict() for m in self.memories]

    @classmethod
    def from_dict(cls, data: List[Dict[str, any]]) -> 'MemoryStore':
        store = cls()
        for mem_data in data:
            store.add(MemoryBlock.from_dict(mem_data))
        return store


class FullERS:
    """
    Enhanced Reconsideration System with low‑rank compression and multi‑petal attention.

    This class couples the PMLLLattice with temporal decay, consensus and
    contradiction detection. When new memories are added, it re‑fits the
    vectorizer and SVD to the full corpus and recomputes embeddings for all
    memories. This is feasible for small numbers of memories but would need to
    be optimised for large‑scale use.
    """

    def __init__(self) -> None:
        self.pmll = PMLLLattice(hidden_dim=HIDDEN_DIM, num_petals=NUM_PETALS)
        self.memory_store = MemoryStore()
        self._corpus: List[str] = []
        self.deferred_queue: deque = deque()

    # ------------------------------------------------------------------
    def _update_embeddings(self) -> None:
        """Recompute embeddings for all memories using the PMLL lattice."""
        if not self._corpus:
            return
        # Fit and transform on full corpus
        embeddings = self.pmll.fit_transform(self._corpus)
        # Assign embeddings
        for i, mem in enumerate(self.memory_store.memories):
            mem.embedding = embeddings[i]

    def add_memory(self, content: str, source_quality: float = 0.8, volatility: float = 0.1) -> int:
        """Add a new memory and recompute embeddings."""
        mem = MemoryBlock(content=content, source_quality=source_quality, volatility=volatility)
        self._corpus.append(content)
        self.memory_store.add(mem)
        # Recompute embeddings for all memories
        self._update_embeddings()
        return len(self.memory_store) - 1

    # ------------------------------------------------------------------
    # Temporal decay and consensus
    def _compute_decay(self, mem: MemoryBlock, current_time: Optional[float] = None) -> float:
        t = current_time or time.time()
        dt = max(0.0, t - mem.timestamp)
        lambda_i = LAMBDA_BASE * (1 + BETA / (1 + mem.source_quality)) * (1 + GAMMA * mem.volatility)
        decay_factor = math.exp(-lambda_i * dt)
        access_factor = 1 + ALPHA * math.log(1 + mem.access_count)
        decayed_conf = mem.confidence * decay_factor * mem.source_quality * access_factor
        return max(0.0, min(decayed_conf, 1.0))

    def _compute_consensus(self, mem_index: int, related: List[Tuple[int, float]], current_time: Optional[float] = None) -> float:
        mem = self.memory_store.memories[mem_index]
        if not related:
            return self._compute_decay(mem, current_time)
        numerator = 0.0
        denominator = 0.0
        t = current_time or time.time()
        for idx, similarity in related:
            other_mem = self.memory_store.memories[idx]
            age_factor = math.exp(-(t - other_mem.timestamp) / 86400.0)
            weight = similarity * age_factor
            agreement = float(np.dot(mem.embedding, other_mem.embedding)) if mem.embedding is not None else 0.0
            other_conf = self._compute_decay(other_mem, current_time)
            numerator += weight * agreement * other_conf
            denominator += weight
        if denominator == 0.0:
            return self._compute_decay(mem, current_time)
        return max(0.0, min(numerator / denominator, 1.0))

    # ------------------------------------------------------------------
    # Contradiction detection
    @staticmethod
    def _detect_contradiction(mem: MemoryBlock, other: MemoryBlock) -> float:
        if mem.embedding is None or other.embedding is None:
            return 0.0
        similarity = float(np.dot(mem.embedding, other.embedding))
        negation_words = {'not', 'no', 'never', 'false', 'wrong', "n't"}
        has_neg_mem = any(word in mem.content.lower() for word in negation_words)
        has_neg_other = any(word in other.content.lower() for word in negation_words)
        if similarity > 0.7 and has_neg_mem != has_neg_other:
            return 0.8
        return (1.0 - similarity) * 0.3

    # ------------------------------------------------------------------
    # Memory reconsideration
    def reconsider_memory(self, index: int) -> Tuple[float, List[float]]:
        mem = self.memory_store.memories[index]
        conf_decay = self._compute_decay(mem)
        related = self.memory_store.find_similar(index)
        consensus = self._compute_consensus(index, related)
        contradictions: List[float] = []
        for i, _ in related:
            other = self.memory_store.memories[i]
            c_score = self._detect_contradiction(mem, other)
            if c_score > CONTRADICT_THRESHOLD:
                contradictions.append(c_score)
                other.status = 'CONTRADICTED'
        penalty = math.exp(-sum(contradictions)) if contradictions else 1.0
        new_conf = conf_decay * (0.5 + 0.5 * consensus) * penalty
        mem.confidence = max(0.0, min(new_conf, 1.0))
        mem.access_count += 1
        return mem.confidence, contradictions

    # ------------------------------------------------------------------
    # Deferred processing
    def defer_memory(self, index: int, initial_score: float = 0.5) -> None:
        mem = self.memory_store.memories[index]
        mem.status = 'DEFERRED'
        self.deferred_queue.append((index, initial_score))

    def reconsider_deferred(self, max_depth: int = 5) -> None:
        if max_depth <= 0 or not self.deferred_queue:
            return
        initial_length = len(self.deferred_queue)
        processed = 0
        while processed < initial_length:
            index, score = self.deferred_queue.popleft()
            new_conf, contradictions = self.reconsider_memory(index)
            new_score = score * new_conf
            mem = self.memory_store.memories[index]
            if contradictions or new_score < 0.5:
                self.deferred_queue.append((index, new_score))
            else:
                mem.status = 'RESOLVED'
            processed += 1
        self.reconsider_deferred(max_depth - 1)

    def query(self, query_text: str, top_k: int = 3, threshold: float = 0.0) -> List[Tuple[float, int, str]]:
        """Query the memory store for similar memories to the query text."""
        if not self._corpus:
            return []
        try:
            query_emb = self.pmll.transform([query_text])[0]
        except:
            return []
        sims = []
        for i, mem in enumerate(self.memory_store.memories):
            if mem.embedding is not None:
                sim = float(np.dot(query_emb, mem.embedding))
                if sim > threshold:
                    sims.append((sim, i, mem.content))
        sims.sort(key=lambda x: x[0], reverse=True)
        return sims[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    def save_state(self, path: str) -> None:
        state = {
            'memories': self.memory_store.to_dict(),
            'corpus': self._corpus,
            'deferred': list(self.deferred_queue),
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        # Save models with pickle
        models = {
            'vectorizer': self.pmll.vectorizer,
            'svd': self.pmll.svd,
            'petals': self.pmll.petals,
            'hidden_dim': self.pmll.hidden_dim,
        }
        pkl_path = path + '.models.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(models, f)

    def load_state(self, path: str) -> None:
        with open(path, 'r') as f:
            state = json.load(f)
        self._corpus = state.get('corpus', [])
        self.memory_store = MemoryStore.from_dict(state.get('memories', []))
        self.deferred_queue = deque(state.get('deferred', []))

        # Load models
        pkl_path = path + '.models.pkl'
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                models = pickle.load(f)
            self.pmll.vectorizer = models['vectorizer']
            self.pmll.svd = models['svd']
            self.pmll.petals = models.get('petals')
            self.pmll.hidden_dim = models['hidden_dim']
        else:
            # Fallback: refit
            if self._corpus:
                _ = self.pmll.fit_transform(self._corpus)
                self.pmll.petals = None  # Will be initialized in fit_transform

        # Compute embeddings
        if self._corpus and hasattr(self.pmll.vectorizer, 'vocabulary_') and self.pmll.vectorizer.vocabulary_:
            embeddings = self.pmll.transform(self._corpus)
            for i, mem in enumerate(self.memory_store.memories):
                mem.embedding = embeddings[i]


def demo() -> None:
    """Demonstration of the fuller ERS with low‑rank compression and attention."""
    path = '/tmp/full_ers_state.json'
    ers = FullERS()
    # Add memories
    i1 = ers.add_memory("Paris is the capital of France", source_quality=0.9, volatility=0.1)
    i2 = ers.add_memory("Paris is the largest city in France", source_quality=0.7, volatility=0.3)
    i3 = ers.add_memory("Lyon is a major city in France", source_quality=0.6, volatility=0.2)
    # Reconsider memories
    c1, contr1 = ers.reconsider_memory(i1)
    print(f"Memory 1 confidence: {c1:.2f}, contradictions: {contr1}")
    c2, contr2 = ers.reconsider_memory(i2)
    print(f"Memory 2 confidence: {c2:.2f}, contradictions: {contr2}")
    # Defer third memory
    ers.defer_memory(i3)
    ers.reconsider_deferred()
    mem3 = ers.memory_store.memories[i3]
    print(f"Memory 3 status: {mem3.status}, confidence: {mem3.confidence:.2f}")
    # Save state
    ers.save_state(path)
    print("State saved to", path)

    # Query
    print("\nQuerying 'capital of France':")
    results = ers.query("capital of France")
    for sim, idx, cont in results:
        print(f"Sim: {sim:.2f}, Content: {cont}")

    # Load state
    ers2 = FullERS()
    ers2.load_state(path)
    print("\nAfter load, query 'capital of France':")
    results2 = ers2.query("capital of France")
    for sim, idx, cont in results2:
        print(f"Sim: {sim:.2f}, Content: {cont}")


if __name__ == '__main__':
    demo()
