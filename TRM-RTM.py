import torch
import torch.nn as nn
import time
import math

# MemoryBlock: Represents an individual memory unit as described in RTM whitepaper
class MemoryBlock:
    def __init__(self, content, embedding, conf=1.0, src="model"):
        self.content = content  # The content/claim
        self.embedding = embedding  # Vector embedding
        self.conf = conf  # Initial confidence [0,1]
        self.t = time.time()  # Creation timestamp
        self.src = src  # Source metadata

# RTMMemory: Implements persistent memory with temporal decay, consensus, and contradiction detection from RTM
class RTMMemory:
    def __init__(self, embed_dim, lambda0=0.01, alpha=0.1, beta=0.1, gamma=0.1, tau_sim=0.7, tau_c=0.5):
        self.memories = []  # List of MemoryBlock
        self.embed_dim = embed_dim
        self.lambda0 = lambda0  # Base decay rate
        self.alpha = alpha  # Access reinforcement
        self.beta = beta  # Source quality weight
        self.gamma = gamma  # Domain volatility weight
        self.tau_sim = tau_sim  # Similarity threshold for related memories
        self.tau_c = tau_c  # Contradiction threshold
        self.access_counts = {}  # Track access for each memory (by index)

    def add_memory(self, content, embedding, conf=1.0, src="model"):
        mem = MemoryBlock(content, embedding, conf, src)
        self.memories.append(mem)
        self.access_counts[len(self.memories) - 1] = 0  # Initialize access count

    def temporal_decay(self, mem, idx):
        t_diff = time.time() - mem.t
        # Simplified enhanced decay (assuming Q_i=1, v_i=0 for demo)
        A_i = self.access_counts.get(idx, 0)
        decay = mem.conf * math.exp(-self.lambda0 * t_diff) * (1 + self.alpha * math.log(1 + A_i))
        return decay

    def find_related(self, query_embedding):
        related = []
        for idx, mem in enumerate(self.memories):
            sim = torch.cosine_similarity(query_embedding, mem.embedding, dim=-1).mean().item()  # Average over batch if needed
            if sim > self.tau_sim:
                self.access_counts[idx] += 1  # Increment access
                related.append((idx, mem))
        return related

    def compute_consensus(self, related):
        if not related:
            return 1.0
        weights = []
        confs = []
        for idx, mem in related:
            w = 1.0  # Simplified weight (can add sim * age_factor)
            conf = self.temporal_decay(mem, idx)
            weights.append(w)
            confs.append(conf)
        consensus = sum(w * c for w, c in zip(weights, confs)) / sum(weights)
        return consensus

    def detect_contradiction(self, query_embedding, related):
        # Simplified: Average "negation score" (random for demo; in practice, use negation detection)
        contradiction = 0.0
        for _, mem in related:
            # Mock negation score based on embedding distance or antonym check
            negation = 0.2  # Placeholder
            sim = torch.cosine_similarity(query_embedding, mem.embedding, dim=-1).mean().item()
            contradiction += sim * negation
        return contradiction / max(1, len(related)) if related else 0.0

    def retrieve(self, query_embedding):
        related = self.find_related(query_embedding)
        consensus = self.compute_consensus(related)
        contradiction = self.detect_contradiction(query_embedding, [m for _, m in related])
        conf = consensus * math.exp(-contradiction)
        if related:
            agg_emb = torch.mean(torch.stack([m.embedding for _, m in related]), dim=0)
        else:
            agg_emb = torch.zeros_like(query_embedding)
        return agg_emb, conf

    def reconsider(self, query_embedding, new_content, new_embedding):
        # Simplified reconsideration: Update confidences and potentially add new memory
        related = self.find_related(query_embedding)
        for idx, mem in related:
            # Penalize if contradiction high
            if self.detect_contradiction(query_embedding, [mem]) > self.tau_c:
                mem.conf *= 0.5  # Reduce confidence
        self.add_memory(new_content, new_embedding)  # Add updated knowledge

# SimpleNet: A flexible MLP for processing concatenated inputs (used in TRM recursion)
class SimpleNet(nn.Module):
    def __init__(self, d_model, max_mult=3):
        super().__init__()
        self.proj = nn.Linear(max_mult * d_model, d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),  # SwiGLU-like activation
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, *inputs):
        cat = torch.cat(inputs, dim=-1)
        proj = self.proj(cat)
        return self.mlp(proj)

# HybridTRMRTM: Hybrid model combining TRM's recursive reasoning with RTM's persistent memory logic loops
class HybridTRMRTM(nn.Module):
    def __init__(self, d_model, num_classes, embed_dim=None):
        super().__init__()
        self.d_model = d_model
        self.net = SimpleNet(d_model)
        self.output_head = nn.Linear(d_model, num_classes)  # For classification tasks (e.g., puzzle output)
        self.q_head = nn.Linear(d_model, 1)  # For halting probability
        self.memory = RTMMemory(embed_dim or d_model)  # RTM memory integration
        # Embedding layer for inputs (if needed for memory queries)
        self.embed = nn.Linear(d_model, self.memory.embed_dim)  # Project to embed_dim if different

    def latent_recursion(self, x, y, z, n=6):
        for _ in range(n):
            z = self.net(x, y, z)
        y = self.net(y, z)
        return y, z

    def deep_recursion(self, x, y, z, n=6, T=3):
        with torch.no_grad():
            for _ in range(T - 1):
                y, z = self.latent_recursion(x, y, z, n)
        y, z = self.latent_recursion(x, y, z, n)
        return (y.detach(), z.detach()), self.output_head(y), self.q_head(y).sigmoid()

    def forward(self, x, query_content=None, query_embedding=None, update_memory=False):
        # Hybrid: Augment x with RTM memory if query provided
        if query_embedding is None and query_content is not None:
            # Mock embedding from content (in practice, use SentenceTransformer or similar)
            query_embedding = self.embed(x)  # Use x as proxy for demo
        if query_embedding is not None:
            mem_emb, conf = self.memory.retrieve(query_embedding)
            x = x + mem_emb * conf  # Augment input with memory

        # TRM recursive reasoning
        y_init = torch.zeros_like(x)
        z_init = torch.zeros_like(x)
        (y, z), y_hat, q_hat = self.deep_recursion(x, y_init, z_init)

        # Optional: Update memory with output (RTM persistent update)
        if update_memory and query_content is not None:
            output_content = f"Output: {y_hat.argmax().item()}"  # Mock content
            output_embedding = self.embed(y)  # Embed output
            self.memory.reconsider(query_embedding, output_content, output_embedding)

        return y_hat, q_hat

# Example usage/demo
if __name__ == "__main__":
    d_model = 128
    num_classes = 10
    model = HybridTRMRTM(d_model, num_classes)

    # Sample input (batch=1, seq=1, dim=d_model)
    x = torch.randn(1, 1, d_model)
    query_embedding = torch.randn(1, 1, d_model)  # Mock query

    # Forward pass with memory augmentation
    y_hat, q_hat = model(x, query_content="Sample query", query_embedding=query_embedding, update_memory=True)
    print("Output:", y_hat)
    print("Halting prob:", q_hat)

    # Add some memories for demo
    model.memory.add_memory("Memory 1", torch.randn(1, 1, d_model))
    model.memory.add_memory("Memory 2", torch.randn(1, 1, d_model))

    # Retrieve and print consensus for a query
    agg_emb, conf = model.memory.retrieve(query_embedding)
    print("Retrieved memory conf:", conf)
```​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​
