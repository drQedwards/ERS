# The Recursive Transformer Model (RTM) — notes for ERS integration

Citation
--------
Josef Kurk Edwards. The Recursive Transformer Model: Architecture, Theory, and Implementation with Persistent Memory Logic Loops. TechRxiv. October 23, 2025.  
DOI: 10.36227/techrxiv.176118936.69886233/v1  
URL: https://www.techrxiv.org/users/856117/articles/1345789-the-recursive-transformer-model-architecture-theory-and-implementation-with-persistent-memory-logic-loops

Funder
------
- Funder Identifier: 100000005  
- Funder Name: U.S. Department of Defense

Overview
--------
This document summarizes the Recursive Transformer Model (RTM) as described in the cited TechRxiv paper and provides actionable guidance for integrating RTM concepts into the ERS (Enhanced Reconsideration System) project.

RTM in one paragraph
--------------------
RTM augments standard transformer architectures with persistent memory "logic loops" that support iterative, multi-pass processing over persistent memory slots. Each recursive pass can read from and write to persistent memory, allowing the model to refine beliefs (embeddings, confidences, textual rewrites) across iterations. The model emphasizes (1) persistent memory structures, (2) looped inference passes (reconsideration), and (3) mechanisms for contradiction detection and consensus-driven rewrites — all core concerns of ERS.

How RTM maps to ERS components
------------------------------
- Persistent memory slots (RTM) -> MemoryLine / MemoryBlock (ERS)
  - Use MemoryBlock fields (id, text, embedding, confidence, timestamps) as described in ERS.
- Logic loops / recursive passes (RTM) -> recursive_loop_check() + reconsider_deferred() (ERS)
  - RTM's iterative passes correspond to ERS's multi-pass reconsideration loops; both support multiple read/write cycles on memory.
- Consensus & contradiction detection (RTM) -> compute_consensus(), detect_contradiction() (ERS)
  - RTM techniques for voting and contradiction scoring can be implemented with ERS heuristics (cosine thresholds, temporal decay, weighted consensus).
- Persistent storage & checkpoints (RTM) -> ers_state.json + lattice_state.safetensors (ERS)
  - RTM's persistent memory maps to ERS JSON for metadata and safetensors for heavy tensor checkpoints.
- Routing & tensor processing (RTM) -> PMLLLattice, XGraphMemory, AttentionFlower (ERS)
  - RTM's internal routing maps to ERS lattice and attention modules; implementable as pluggable modules.

Recommended additions / changes to ERS to better support RTM-style experiments
-------------------------------------------------------------------------------
1. Iteration-control config
   - Add explicit control over number of recursive passes, per-memory-slot max-resolves, and early-stopping criteria (consensus threshold, confidence delta).
2. Pass-level instrumentation
   - Log per-pass deltas: embedding shifts (cosine), confidence changes, rewrite counts. Enables ablation and convergence tracking.
3. Memory-versioning and provenance
   - Maintain per-slot version history (or a lightweight append-only chain) with SHA-256 references and optional diffs to support rollbacks and provenance analysis.
4. Hookable rewrite policies
   - Provide interface for 'rewrite_proposer' functions (LLM-guided or deterministic) so a pass may suggest textual changes and an acceptance policy decides whether to commit them.
5. Bench and example notebooks
   - Add a small RTM/ERS notebook demonstrating a simple two-pass (double) recursive loop where an initial assertion is refined twice using KG/Mem0 lookups and consensus.

Simple recommended experiment (double-pass RTM)
-----------------------------------------------
1. Seed memory with short factual assertions and a contradictory assertion.
2. Run a double-pass pipeline:
   - Pass 1: indexing & local consensus (neighbour retrieval + embedding refinement).
   - Pass 2: contradiction detection + rewrite proposal + acceptance via consensus.
3. Measure convergence: number of rewritten slots, confidence increases, contradiction scores lowered.
4. Visualize memory graph evolution with timestamps.

API and configuration snippets
------------------------------
- Configuration options (suggested additions)
  - rtm:
    - passes: 2
    - early_stop_cosine_delta: 0.002
    - max_rewrites_per_slot: 1
    - decay_alpha: 0.95

- Example high-level flow (pseudocode)
  - for pass in 1..passes:
      for slot in memory_slots:
          decay(slot)
          neighbours = find_related(slot.embedding)
          consensus = compute_consensus(neighbours)
          if detect_contradiction(slot, neighbours):
              proposal = propose_rewrite(slot, consensus)
              if acceptance_policy(proposal, slot, neighbours):
                  commit_rewrite(slot, proposal)
          slot.embedding = reembed(slot.text)  # or refinement via lattice

Licensing & Attribution
-----------------------
- When using or adapting the RTM design or text from the TechRxiv paper, include the citation above in documentation and any derivative works (as required by good scholarly practice).
- ERS remains MIT-licensed; any contributions inspired by RTM should follow repository contribution/licensing policies and preserve the citation.

Appendix A — Practical notes for implementers
---------------------------------------------
- Embedding stubs: when experimenting without an LLM or sentence-transformer, deterministic pseudo-embeddings (hash-based) are useful for validating loop behavior.
- Local KG (Graphiti) vs remote KG: for fast experiments use an in-memory Graphiti or lightweight Neo4j instance; ensure reconcilers are resilient to KG timeouts.
- Performance: multiple passes increase compute; keep tensor checkpointing (safetensors) and state roll-forward minimal to reduce overhead.
- Safety: add policy checks for any automated rewrite gating for safety and fidelity before committing changes to persistent memory.

Acknowledgements
----------------
This RTM integration note was prepared for the ERS project and references: Josef Kurk Edwards. The Recursive Transformer Model: Architecture, Theory, and Implementation with Persistent Memory Logic Loops. TechRxiv. October 23, 2025. DOI: 10.36227/techrxiv.176118936.69886233/v1

If you'd like, I can:
- extract key algorithm pseudocode from the paper and encode it as ERS-compatible functions (Python),
- produce a Jupyter notebook with a two-pass experiment using ERS mock components, or
- create a ready-to-commit PR that adds RTM config options and an example pipeline to the repo.
