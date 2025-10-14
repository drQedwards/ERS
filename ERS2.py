"""
Enhanced Reconsideration System (ERS) - Production Implementation
A stateful memory management system for AI agents with temporal decay,
consensus validation, and knowledge graph integration.
"""

import asyncio
import json
import time
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
import logging

from sentence_transformers import SentenceTransformer
from safetensors.torch import save_file, load_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
SIM_THRESHOLD = 0.7
CONTRADICT_THRESHOLD = 0.5
MAX_RECURSION_DEPTH = 5

# Temporal decay parameters
LAMBDA_BASE = 0.001
ALPHA = 0.1
BETA = 0.5
GAMMA = 0.2

# State persistence
STATE_FILE = 'ers_state.json'
LATTICE_FILE = 'ers_lattice.safetensors'


class MemoryBlock:
    """Core memory unit with integrity hashing and temporal metadata."""
    
    def __init__(self, content: str, source_quality: float = 0.8, 
                 volatility: float = 0.1):
        self.content = content
        self.confidence = 1.0
        self.timestamp = time.time()
        self.source_quality = np.clip(source_quality, 0, 1)
        self.volatility = np.clip(volatility, 0, 1)
        self.access_count = 0
        self.embedding: Optional[np.ndarray] = None
        self.prev_hash: Optional[str] = None
        self.hash = self._compute_hash()
        self.status = 'ACTIVE'  # ACTIVE, DEFERRED, RESOLVED, CONTRADICTED

    def _compute_hash(self) -> str:
        """SHA-256 hash for integrity verification."""
        data = f"{self.content}{self.timestamp}{self.confidence}".encode()
        return hashlib.sha256(data).hexdigest()

    async def get_embedding(self, embedder: SentenceTransformer) -> np.ndarray:
        """Lazy-load embedding with caching."""
        if self.embedding is None:
            self.embedding = embedder.encode(
                self.content, 
                normalize_embeddings=True
            )
        return self.embedding

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for persistence."""
        return {
            'content': self.content,
            'confidence': float(self.confidence),
            'timestamp': self.timestamp,
            'source_quality': float(self.source_quality),
            'volatility': float(self.volatility),
            'access_count': self.access_count,
            'prev_hash': self.prev_hash,
            'hash': self.hash,
            'status': self.status,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryBlock':
        """Deserialize from dict."""
        obj = cls(
            data['content'],
            data['source_quality'],
            data['volatility']
        )
        obj.confidence = data['confidence']
        obj.timestamp = data['timestamp']
        obj.access_count = data['access_count']
        obj.prev_hash = data['prev_hash']
        obj.hash = data['hash']
        obj.status = data.get('status', 'ACTIVE')
        if data.get('embedding'):
            obj.embedding = np.array(data['embedding'])
        return obj


class AttentionFlower(torch.nn.Module):
    """Multi-petal attention for tensor processing."""
    
    def __init__(self, num_petals: int = 8, hidden_dim: int = 384):
        super().__init__()
        self.num_petals = num_petals
        self.petals = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_petals)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention across petals."""
        outputs = []
        for petal in self.petals:
            out = F.relu(petal(x))
            outputs.append(out)
        return torch.mean(torch.stack(outputs), dim=0)


class PMLLLattice:
    """PMLL Lattice for memory compression and efficient routing."""
    
    def __init__(self, hidden_dim: int = 384, rank: int = 64):
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.attention_flower = AttentionFlower(
            num_petals=8,
            hidden_dim=hidden_dim
        )
        self.state: Dict[str, torch.Tensor] = {}

    async def process_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Process embedding through attention flower."""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        processed = self.attention_flower(embedding)
        return F.normalize(processed, p=2, dim=-1)

    def save_checkpoint(self, path: str) -> None:
        """Save PMLL state using safetensors."""
        if self.state:
            save_file(self.state, path)
            logger.info(f"Lattice checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load PMLL state from safetensors."""
        if Path(path).exists():
            self.state = load_file(path)
            logger.info(f"Lattice checkpoint loaded: {path}")


class MemoryStore:
    """In-memory storage for memory blocks with similarity indexing."""
    
    def __init__(self):
        self.blocks: Dict[str, MemoryBlock] = {}
        self.embeddings: Dict[str, np.ndarray] = {}

    def add(self, block: MemoryBlock) -> None:
        """Add memory block to store."""
        self.blocks[block.hash] = block
        if block.embedding is not None:
            self.embeddings[block.hash] = block.embedding

    def get(self, hash_id: str) -> Optional[MemoryBlock]:
        """Retrieve memory block by hash."""
        return self.blocks.get(hash_id)

    def find_similar(self, embedding: np.ndarray, 
                    threshold: float = SIM_THRESHOLD) -> List[Tuple[str, float]]:
        """Find similar memories using cosine similarity."""
        if len(self.embeddings) == 0:
            return []
        
        similar = []
        for hash_id, stored_emb in self.embeddings.items():
            sim = np.dot(embedding, stored_emb)
            if sim > threshold:
                similar.append((hash_id, float(sim)))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize store for persistence."""
        return {
            hash_id: block.to_dict()
            for hash_id, block in self.blocks.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryStore':
        """Deserialize from dict."""
        store = cls()
        for hash_id, block_data in data.items():
            block = MemoryBlock.from_dict(block_data)
            store.add(block)
        return store


class EnhancedReconsiderationSystem:
    """Main orchestration system for memory management and reconsideration."""
    
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(embedding_model)
        self.memory_store = MemoryStore()
        self.pmll = PMLLLattice(hidden_dim=384, rank=64)
        
        # Deferred processing queue
        self.deferred_queue: deque = deque()
        
        # Blockchain-like memory chain
        self.blockchain_head: Optional[MemoryBlock] = None
        
        # Load persistent state
        self._load_state()
        logger.info("ERS initialized")

    async def add_memory(self, content: str, source_quality: float = 0.8,
                        volatility: float = 0.1) -> str:
        """Add new memory to the system."""
        mem = MemoryBlock(content, source_quality, volatility)
        mem.prev_hash = self.blockchain_head.hash if self.blockchain_head else None
        
        # Compute embedding
        embedding = await mem.get_embedding(self.embedder)
        
        # Process through PMLL
        emb_tensor = torch.from_numpy(embedding).float()
        processed = await self.pmll.process_embedding(emb_tensor)
        mem.embedding = processed.numpy()
        
        # Add to store
        self.blockchain_head = mem
        self.memory_store.add(mem)
        
        logger.info(f"Memory added: {mem.hash[:8]}... (confidence: {mem.confidence:.2f})")
        return mem.hash

    async def temporal_decay(self, mem: MemoryBlock, 
                            t: Optional[float] = None) -> float:
        """Compute time-decayed confidence."""
        t = t or time.time()
        dt = max(0, t - mem.timestamp)
        
        # Adaptive decay rate
        lambda_i = LAMBDA_BASE * (
            1 + BETA / (1 + mem.source_quality)
        ) * (1 + GAMMA * mem.volatility)
        
        # Exponential decay
        decay_factor = np.exp(-lambda_i * dt)
        
        # Access reinforcement (mimics memory consolidation)
        access_factor = 1 + ALPHA * np.log(1 + mem.access_count)
        
        decayed_conf = (mem.confidence * decay_factor * 
                       mem.source_quality * access_factor)
        
        return float(np.clip(decayed_conf, 0, 1))

    async def find_related(self, mem: MemoryBlock) -> List[Tuple[str, float]]:
        """Find memories related to given memory."""
        if mem.embedding is None:
            return []
        
        related = self.memory_store.find_similar(mem.embedding, SIM_THRESHOLD)
        # Exclude self
        return [(h, sim) for h, sim in related if h != mem.hash]

    async def compute_consensus(self, mem: MemoryBlock,
                               related: List[Tuple[str, float]]) -> float:
        """Compute consensus score from related memories."""
        if not related:
            return await self.temporal_decay(mem)
        
        numerator = 0.0
        denominator = 0.0
        
        for other_hash, similarity in related:
            other_mem = self.memory_store.get(other_hash)
            if other_mem is None:
                continue
            
            other_conf = await self.temporal_decay(other_mem)
            age_factor = np.exp(-(time.time() - other_mem.timestamp) / 86400.0)
            weight = similarity * age_factor
            
            # Dynamic agreement via embedding alignment
            if mem.embedding is not None and other_mem.embedding is not None:
                agreement = np.dot(mem.embedding, other_mem.embedding)
            else:
                agreement = 0.5
            
            numerator += weight * agreement * other_conf
            denominator += weight
        
        if denominator > 0:
            return float(np.clip(numerator / denominator, 0, 1))
        else:
            return await self.temporal_decay(mem)

    async def detect_contradiction(self, mem1: MemoryBlock,
                                  mem2: MemoryBlock) -> float:
        """Detect contradiction between two memories."""
        if mem1.embedding is None or mem2.embedding is None:
            return 0.0
        
        # Semantic similarity
        sim = float(np.dot(mem1.embedding, mem2.embedding))
        
        # Simple negation detection (check for common negation keywords)
        negation_keywords = {'not', 'no', 'never', 'false', 'wrong'}
        has_negation_1 = any(kw in mem1.content.lower() for kw in negation_keywords)
        has_negation_2 = any(kw in mem2.content.lower() for kw in negation_keywords)
        
        # High similarity + opposite negation patterns = contradiction
        if has_negation_1 != has_negation_2 and sim > 0.7:
            contradiction_score = 0.8
        else:
            contradiction_score = max(0, 1 - sim) * 0.3
        
        return float(np.clip(contradiction_score, 0, 1))

    async def reconsider_memory(self, mem: MemoryBlock) -> Tuple[float, List[float]]:
        """Core reconsideration logic: decay, consensus, contradiction detection."""
        conf_decay = await self.temporal_decay(mem)
        related = await self.find_related(mem)
        consensus = await self.compute_consensus(mem, related)
        
        contradictions = []
        for related_hash, _ in related:
            related_mem = self.memory_store.get(related_hash)
            if related_mem:
                contradict = await self.detect_contradiction(mem, related_mem)
                if contradict > CONTRADICT_THRESHOLD:
                    contradictions.append(contradict)
                    related_mem.status = 'CONTRADICTED'
        
        # Final confidence: decay × consensus × contradiction penalty
        penalty = np.exp(-sum(contradictions)) if contradictions else 1.0
        new_conf = conf_decay * (0.5 + 0.5 * consensus) * penalty
        
        mem.confidence = float(np.clip(new_conf, 0, 1))
        mem.access_count += 1
        
        logger.info(
            f"Memory reconsidered: {mem.hash[:8]}... "
            f"(conf: {mem.confidence:.2f}, contradictions: {len(contradictions)})"
        )
        
        return mem.confidence, contradictions

    def defer_memory(self, mem: MemoryBlock, initial_score: float = 0.5) -> None:
        """Add memory to deferred processing queue."""
        self.deferred_queue.append((mem, initial_score))
        mem.status = 'DEFERRED'
        logger.info(f"Memory deferred: {mem.hash[:8]}...")

    async def reconsider_deferred(self, max_depth: int = MAX_RECURSION_DEPTH) -> None:
        """Recursively process deferred memories."""
        if max_depth <= 0 or len(self.deferred_queue) == 0:
            return
        
        queue_size = len(self.deferred_queue)
        processed = 0
        
        while processed < queue_size:
            mem, score = self.deferred_queue.popleft()
            new_conf, contradicts = await self.reconsider_memory(mem)
            new_score = score * new_conf
            
            if len(contradicts) > 0 or new_score < 0.5:
                # Re-defer if still problematic
                self.deferred_queue.append((mem, new_score))
            else:
                mem.status = 'RESOLVED'
                logger.info(f"Memory resolved: {mem.hash[:8]}...")
            
            processed += 1
        
        # Recursive call with reduced depth
        await self.reconsider_deferred(max_depth - 1)

    def _save_state(self) -> None:
        """Save system state to JSON and safetensors."""
        state = {
            'deferred_queue': [
                {**mem.to_dict(), 'score': score}
                for mem, score in self.deferred_queue
            ],
            'memory_store': self.memory_store.to_dict(),
            'blockchain_head': self.blockchain_head.to_dict() 
                              if self.blockchain_head else None
        }
        
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.pmll.save_checkpoint(LATTICE_FILE)
        logger.info("State saved to disk")

    def _load_state(self) -> None:
        """Load system state from JSON and safetensors."""
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            
            # Restore memory store
            self.memory_store = MemoryStore.from_dict(state.get('memory_store', {}))
            
            # Restore deferred queue
            self.deferred_queue = deque()
            for item in state.get('deferred_queue', []):
                score = item.pop('score', 0.5)
                mem = MemoryBlock.from_dict(item)
                self.deferred_queue.append((mem, score))
            
            # Restore blockchain head
            if state.get('blockchain_head'):
                self.blockchain_head = MemoryBlock.from_dict(state['blockchain_head'])
            
            # Load PMLL checkpoint
            self.pmll.load_checkpoint(LATTICE_FILE)
            
            logger.info(f"State loaded: {len(self.memory_store.blocks)} memories")
        except FileNotFoundError:
            logger.info("No state file found - starting fresh")
        except Exception as e:
            logger.error(f"Error loading state: {e}")

    async def close(self) -> None:
        """Clean shutdown - saves state."""
        self._save_state()
        logger.info("ERS closed")


# Example usage
async def main():
    ers = EnhancedReconsiderationSystem()
    
    # Add some memories
    hash1 = await ers.add_memory(
        "Paris is the capital of France",
        source_quality=0.9,
        volatility=0.1
    )
    
    hash2 = await ers.add_memory(
        "Paris is the largest city in France",
        source_quality=0.7,
        volatility=0.3
    )
    
    # Retrieve and reconsider
    mem1 = ers.memory_store.get(hash1)
    mem2 = ers.memory_store.get(hash2)
    
    if mem1 and mem2:
        # Check for contradiction
        contradict_score = await ers.detect_contradiction(mem1, mem2)
        print(f"Contradiction score: {contradict_score:.2f}")
        
        # Reconsider with consensus
        conf, contrs = await ers.reconsider_memory(mem1)
        print(f"Reconsidered confidence: {conf:.2f}")
    
    # Deferred processing example
    mem3 = MemoryBlock("Lyon is a major city in France", 0.6, 0.2)
    ers.memory_store.add(mem3)
    ers.defer_memory(mem3)
    await ers.reconsider_deferred()
    
    await ers.close()


if __name__ == "__main__":
    asyncio.run(main())
