# ERS — Enhanced Reconsideration System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

A production-ready Python library turning stateless AI agents into stateful ones by providing recursive reconsideration, knowledge-graph backed memory, and robust tensor/embedding workflows (PMLL-inspired).

Table of contents
- Overview
- Key features
- Architecture & core components
- Installation
- Quick start
- Configuration & persistence
- API summary
- Roadmap
- Contributing
- License

Overview
--------
ERS (Enhanced Reconsideration System) is a memory-management and reconsideration framework for AI systems. It combines:
- persistent memory slots and deferred reconsideration queues,
- embedding & tensor processing via a PMLL-like lattice and multi-petal attention,
- knowledge-graph integration (Graphiti) and ID-based memory store (Mem0),
- recursive loops for multi-pass validation (reconsideration),
- consensus, temporal decay and contradiction detection for self-correction,
- state persistence (JSON + safetensors) for continuity across runs.

Designed for production use, ERS focuses on async operation, thread-safety, and modular integration with existing LLM-based stacks.

Key features
------------
- Stateful transformation of stateless models: load/save memory, queues and lattice state.
- ERSPromise: chainable async promises connected to MemoryBlocks for deferred resolution.
- Deferred queue + MemoryLine: priority & circular buffer memory structures with thread locks.
- Temporal decay, consensus voting, and contradiction detection (mathematical heuristics).
- PMLL Lattice + X-Graph routing + AttentionFlower: dynamic tensor routing and embedding refinement.
- Graphiti & Mem0 integration for episode addition, querying, and local/global rewrites.
- LangChain compatibility for conversational KG memory and agent-based grounding.
- Safe persistence: JSON for structural state and safetensors for lattice/tensor checkpoints.
- Blockchain-style SHA-256 hashing for integrity of memory-block chains.

Architecture & core components
------------------------------
High-level components:
- MemoryBlock — content unit with IDs, confidence, timestamps, embedding and hash. Serializable.
- ERSPromise — async future wrapper linking memory blocks in chainable promises.
- MemoryLine — slotted circular buffer for active memory slots and operations (push/pull/move).
- PMLLLattice — tensor processing core; supports hooks, attention modules, checkpointing via safetensors.
- XGraphMemory — routing/compression layer for tensor paths.
- AttentionFlower — multi-petal attention module for richer embedding transformations.
- PMLL — coordinator for Graphiti, Mem0 and lattice flows (episode addition, querying, rewriting).
- EnhancedReconsiderationSystem — orchestrator tying KG, memory, lattice, promises, queues and loops.

Core algorithms & behaviors
- temporal_decay(mem, now) — decays confidence using adaptive rates (configurable).
- find_related(embedding) — KG + Mem0 nearest-neighbour style retrieval for context & consensus.
- compute_consensus(candidates) — weighted voting and agreement scores.
- detect_contradiction(block, neighbours) — semantic/temporal/entity contradiction scoring.
- reconsider_memory(block) — main reconsideration flow: decay → consensus → contradiction → optional rewrite & re-grounding → embedding re-process.
- defer_memory(block, score) — enqueue for later reconsideration with score-based scheduling.
- recursive_loop_check() — iterates slots and queue using PMLL-inspired while-loops for multi-pass reconsideration.
- chain_promises(p1, p2, ...) — links promises and concatenates embedding tails for downstream inference.

Installation
------------
Requirements
- Python 3.8+
- Core libs: asyncio, numpy, collections, threading, hashlib, json, datetime, typing
- ML libs: torch, sentence-transformers
- Persistence: safetensors
- KG & memory: mem0, graphiti-core
- LangChain and community connectors for agent-based grounding

Install (example):
pip install numpy sentence-transformers torch safetensors mem0-ai graphiti-core langchain langchain-community langchain-openai

Note: replace langchain-openai with the provider package that matches your LLM.

Quick start
-----------
Minimal async example:

```python
import asyncio
from ERS import EnhancedReconsiderationSystem, MemoryBlock, ERSPromise

async def main():
    ers = EnhancedReconsiderationSystem()  # loads saved state if present

    await ers.add_memory("Paris is the capital of France")
    mem1 = MemoryBlock("Paris is the capital of France")

    await ers.add_memory("Paris is the largest city in France")
    mem2 = MemoryBlock("Paris is the largest city in France")

    ers.chain_promises(ERSPromise(mem1), ERSPromise(mem2))

    ers.defer_memory(mem1)
    ers.defer_memory(mem2)

    await ers.reconsider_deferred()
    await ers.recursive_loop_check()
    await ers.close()  # saves state/backups

asyncio.run(main())
```

Configuration & persistence
---------------------------
- State files:
  - ers_state.json — stores queues, memory-line slots, head pointers and non-tensor data.
  - lattice_state.safetensors — contains lattice tensors and checkpoints.
- KG: Graphiti (Neo4j recommended). Configure connection URIs and credentials in the PMLL/PMLLConfig class or via environment variables.
- Mem0: configure IDs and storage backends according to mem0 docs.
- Checkpointing is safe-tensor based for speed and safety in production.

API summary
-----------
(Condensed; see code docstrings for full details)
- EnhancedReconsiderationSystem
  - add_memory(text, metadata=None) → adds MemoryBlock, processes embeddings, adds to KG/Mem0
  - defer_memory(memory_block, score=None) → defers for reconsideration
  - reconsider_deferred() → processes the deferred queue (async)
  - recursive_loop_check() → performs multi-pass slot reconsideration (async)
  - chain_promises(*promises) → link promise chain
  - close() → saves JSON + safetensors

- MemoryBlock
  - to_dict(), from_dict() — serialization helpers
  - fields: id, text, embedding, confidence, created_at, updated_at, sha256_hash, kg_id, mem0_id

- ERSPromise
  - resolve(), then(next_promise) — promise chaining helpers

Development & testing
---------------------
- Use virtual environments and pin dependencies when deploying.
- Tests should mock KG & Mem0 interfaces or run a local Neo4j + mem0 test instance.
- Lattice operations can be checkpointed with safetensors for regression testing.

Roadmap
-------
- Replace stub callbacks with LLM-guided embeddings & reasoning (gRPC/HTTP options).
- Visualize KG evolution and ERS revision history (temporal graph animation).
- Define “axiom nodes” and lucidity flags for system self-awareness.
- Add telos/ethical filter layer for narrative steering and safety.

Contributing
------------
Contributions are welcome. Please:
1. Fork the repo
2. Create feature branches
3. Add tests and update docs
4. Open PRs with clear descriptions and rationale

If you contribute embedding backends, reasoning modules, or new ERS heuristics, document them and keep the double-loop semantics in mind.

License
-------
MIT — see LICENSE file.

Acknowledgements
----------------
Inspired by the "Enhanced Reconsideration" white paper and the work of cognitive-inspired memory & KG systems. Authored by Dr. Q (Dr. Q / Phoenix Harmonic).

If you'd like, I can:
- generate an expanded API reference section with docstring extraction,
- create example notebooks demonstrating KG operations,
- or push a ready-to-commit README directly to your repository.
