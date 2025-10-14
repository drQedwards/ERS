# ERS
Enhanced Reconsideration Library in Python for recursive Q- PMLL memory looping using Graphiti's Knowledge Graph to rewrite and automate knowledge base updating

# Enhanced Reconsideration System (ERS)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

## Overview

The Enhanced Reconsideration System (ERS) is a production-grade AI memory management library designed to transform stateless AI agents into stateful ones. It integrates asynchronous promise chains, knowledge graph refinement using Graphiti and Mem0, temporal decay mechanisms, consensus validation, and contradiction detection, drawing from advanced AI memory architectures. ERS enables AI systems to reconsider deferred memory blocks recursively, update embeddings with fresh inferences from search grounding, and persist state using safe tensors and JSON serialization.

Key innovations:
- **Stateful Transformation**: On initialization, ERS loads persistent state from JSON and safe tensors, making any AI model (e.g., Grok, Gemini, ChatGPT, or non-transformer models) stateful by maintaining memory queues, chains, and embeddings across sessions.
- **Recursive Looping and Reconsideration**: Implements PMLL-inspired recursive while-loops for multi-pass validation, integrating blockchain-like hashing for integrity.
- **Knowledge Graph Integration**: Uses Graphiti for temporal KG management, Mem0 for high-level memory ops, and LangChain for conversational KG memory and agent-based search grounding.
- **Tensor Processing**: Employs PMLL Lattice, X-Graph routing, and AttentionFlower for dynamic embedding refinement and concatenation of promise tails.
- **Production Readiness**: Async operations with `asyncio`, thread-safety via locks, error handling, and optimizations for real-time AI applications.

ERS is inspired by the "Enhanced Reconsideration System" white paper, addressing "nostalgic incorrectness" in AI memory through temporal awareness, consensus, and self-correction.

## Features

- **Asynchronous Promises**: Chainable `ERSPromise` objects for deferred resolutions, integrated with memory blocks.
- **Deferred Queue Management**: A deque for pending memory reconsiderations, with score-based re-deferral.
- **Memory Line Slots**: Circular buffer for pushing/pulling/moving promises or LangChain memories, with thread-safe operations.
- **Temporal Decay and Validation**: Mathematical models for confidence decay (Eqs. 4-6 from inspiration), consensus (Eq. 10), and contradiction detection (Eqs. 11-13).
- **PMLL Integration**: Full lattice processing for tensor routing through X-Graph, custom hooks, and multi-petal attention.
- **Graphiti and Mem0**: For KG episode addition, querying, invalidation, and updates; Mem0 handles ID-based memory management.
- **LangChain Integration**: ConversationKGMemory for KG-backed chains, agent for search tools to ground updates with fresh data.
- **State Persistence**: JSON for non-tensor data (queues, blocks), safe tensors for lattice checkpoints.
- **Dynamic Stubs**: Embeddings and scores computed on-the-fly, avoiding static values for negation, agreement, etc.
- **Blockchain Hashing**: Memory blocks linked via SHA-256 hashes for integrity during chains.

## Dependencies

- Python 3.8+
- Core Libraries: `asyncio`, `numpy`, `sentence_transformers`, `collections`, `threading`, `hashlib`, `json`, `datetime`, `typing`
- ML/Embedding: `torch`, `torch.nn`, `torch.nn.functional`
- Persistence: `safetensors`
- Knowledge Graphs/Memory: `mem0`, `graphiti-core`
- LangChain: `langchain`, `langchain-community`, `langchain-openai` (replace OpenAI with your LLM provider)
- Database: Neo4j (for Graphiti; configure URI, user, password)

Install via: 

pip install numpy sentence-transformers torch safetensors mem0-ai graphiti-core langchain langchain-community langchain-openai

Setup Neo4j database and configure credentials in `PMLL` class.

## Architecture

### Core Classes

1. **MemoryBlock**: Represents individual memory units with content, confidence, timestamps, embeddings, hashes, and IDs for Graphiti/Mem0. Supports serialization via `to_dict`/`from_dict`.

2. **ERSPromise**: Async future for promise chaining, with resolution callbacks and next links. Integrates with memory blocks.

3. **MemoryLine**: Slotted circular buffer for managing promises/LangChain memories. Supports push/pull/move operations with locking.

4. **PMLLLattice**: Processes tensors through hooks, attention flower, and graph routing. Saves/loads checkpoints via safe tensors.

5. **XGraphMemory**: Routes tensors via optimal paths with compression.

6. **AttentionFlower**: Multi-petal attention module for enhanced tensor processing.

7. **MyCustomHook**: Custom hook for normalization and model application.

8. **PMLL**: Wrapper for Graphiti, Mem0, and lattice. Handles episode addition, queries, rewrites, and initialization.

9. **EnhancedReconsiderationSystem**: Main class orchestrating everything. Initializes components, loads/saves state, adds/reconsiders memories, defers queues, loops checks, chains promises with tail concatenation.

### Key Methods

- **add_memory**: Creates block, adds to KG/Mem0, processes embedding via lattice.
- **temporal_decay**: Computes decayed confidence using adaptive rates.
- **find_related**: Queries Graphiti/Mem0 for similar memories via embeddings.
- **compute_consensus**: Weighted voting with dynamic agreement.
- **detect_contradiction**: Semantic/temporal/entity scores with dynamic negation.
- **reconsider_memory**: Core logic: decay, consensus, contradiction detection, rewriting, search grounding, lattice/hook processing, X-Graph routing.
- **defer_memory**: Adds to queue with Mem0 integration.
- **reconsider_deferred**: Recursive while-loop for queue processing.
- **recursive_loop_check**: Recursive iteration over slots for validation/reconsideration.
- **chain_promises**: Links promises, concatenates embedding tails.
- **_load_state / _save_state**: JSON for structures, safe tensors for lattice.

### State Management

- On init: Loads from `ers_state.json` (queues, slots, head) and `lattice_state.safetensors` (PMLL tensors).
- On close: Saves updated state, ensuring statefulness across runs for any AI model.

## Usage

See the example in `ERS.py`:

```python
async def main():
    ers = EnhancedReconsiderationSystem()  # Loads state if exists
    await ers.add_memory("Paris is the capital of France")
    mem1 = MemoryBlock("Paris is the capital of France")
    await ers.add_memory("Paris is the largest city in France")
    mem2 = MemoryBlock("Paris is the largest city in France")
    ers.chain_promises(ERSPromise(mem1), ERSPromise(mem2))
    ers.defer_memory(mem1)
    ers.defer_memory(mem2)
    await ers.reconsider_deferred()
    await ers.recursive_loop_check()
    await ers.close()  # Saves state
