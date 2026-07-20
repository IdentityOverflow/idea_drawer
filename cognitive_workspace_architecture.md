---
title: Cognitive workspace architecture for durable local agents
---

## Objective

Build an OpenClaw fork architecture that preserves long-horizon coherence at a
32k target context window without relying on transcript stuffing or routine
compaction.

The goal is not to make the model "remember more text". The goal is to present
the model with the right living mental scene for the next turn.

## Design Thesis

Context is not memory.

A context window is closer to a conscious workspace: a narrow, active field
where current goals, entities, commitments, retrieved facts, and possible
actions are bound together. Durable memory lives outside that workspace as
structured state, indexed episodes, and provenance-backed artifacts.

The system should therefore treat context as a cognitive architecture problem,
not a compression problem.

- Working context is for control, not storage.
- The active workspace stays small, intentional, and state-first.
- Most information is latent, indexed, and cue-retrievable.
- Heavy capability detail is loaded only after a route selects it.
- Deliberation happens in bounded ephemeral threads.
- Durable continuity is maintained as structured memory, not chronological
  transcript bulk.

The core behavior to replicate is human-like coherence with small working
memory: immediate pruning of irrelevant detail, durable tracking of concepts and
commitments, and cue-driven retrieval when old detail becomes useful again.

## Core Principle

The model should not remember the conversation.

It should perceive a continuously reconstructed cognitive scene assembled from:

- Current user input.
- Active goals and obligations.
- Structured semantic state.
- Relevant episodic traces.
- Selected evidence and tool affordances.
- A small amount of recent conversational texture.

Chronological transcript is fallback evidence, not the default substrate.

## Cognitive Layers

### 1) Conscious Workspace

The live model context for the Executive.

Purpose:

- Maintain the active scene for the next response or action.
- Bind user intent, current goals, relevant entities, commitments, and selected
  evidence.
- Keep the Executive oriented without replaying history.

Includes:

- Core system identity and runtime policy.
- Current user intent and active thread state.
- Active goals, open loops, and pending commitments.
- Relevant entities and project state.
- Retrieved memory shards with provenance.
- Recent turns only when conversational texture matters.
- Compact capability digest and selected tool affordances.

Excludes by default:

- Full tool schema catalogs.
- Deep historical transcript blocks.
- Verbose tool payload archives.
- Worker internal logs.
- Unrelated memories.

Default budget target:

- 10k-16k steady-state.
- Hard preference to stay below 22k during routine turns.
- 32k is the operating ceiling, not the normal target.

Rule:

- Every item in working context must justify why it is active now.

### 2) Semantic World Model

Durable structured beliefs about the user, projects, systems, preferences, and
decisions.

Purpose:

- Preserve stable meaning independent of transcript order.
- Let the Executive reconstruct the active scene from durable concepts and
  relationships.
- Avoid treating old wording as more important than current state.

Suggested record kinds:

- fact: stable claim believed to be true.
- preference: explicit or repeated user preference.
- principle: design value, operating style, or philosophical stance.
- entity: person, project, system, file, module, model, artifact, channel.
- relationship: typed link between entities.
- decision: chosen direction plus rationale.
- constraint: hard environmental, policy, technical, or user constraint.

Requirements:

- Every stored belief carries provenance.
- Contradictions are tracked instead of blindly overwritten.
- User corrections supersede weaker inferred beliefs.
- Stale beliefs can decay or become inactive without being deleted.

### 3) Episodic Index

A navigable event map, not raw conversation history.

Purpose:

- Preserve what happened without replaying every sentence.
- Support cue-driven retrieval: "that earlier discussion where we decided X
  because Y".
- Provide provenance pointers back to transcript/tool artifacts when exact
  detail is needed.

Episode fields:

- event summary.
- participants and source channel.
- topic tags.
- entities mentioned.
- decisions made.
- unresolved questions.
- commitments opened or closed.
- artifact references.
- time/order metadata.
- provenance pointers.

Rule:

- Episodes are retrieval handles. They are not normally loaded into the
  workspace unless the current turn activates them.

### 4) Commitment Ledger

First-class memory for promises, obligations, open loops, and expected
follow-ups.

Purpose:

- Preserve conversational trust and task continuity.
- Keep promises and pending work active even when recent turns move elsewhere.

Commitment fields:

- actor: who committed.
- target: who receives or depends on the commitment.
- statement: promised action or maintained state.
- trigger: condition that makes it relevant.
- status: open | blocked | done | dropped.
- due/expected timing when known.
- next action.
- provenance.

Rule:

- Relevant open commitments have higher workspace priority than ordinary
  episodic memory.

### 5) Procedural and Capability Memory

Compact knowledge of what the system can do and when to expose deeper tool
detail.

Purpose:

- Avoid loading full tool schemas into the Executive by default.
- Keep the Executive aware of affordances rather than implementation detail.
- Load scoped schemas only for selected capabilities.

Workspace form:

- capability digest.
- active constraints.
- selected route.
- minimal safe tool pack.
- escalation/fallback policy.

Rule:

- Full schemas are loaded after routing, not before.

## Main Components

### 1) Executive Self

Primary conversation agent.

Responsibilities:

- Own user relationship, commitments, and final responses.
- Operate on the conscious workspace.
- Run a fast path for routine turns.
- Trigger retrieval, verification, or delegation only when needed.
- Preserve final response quality and conversational continuity.

Non-goals:

- Store raw history.
- Inspect every tool every turn.
- Perform all deep synthesis in the primary workspace.

### 2) Memory Steward

Persistent context curator and workspace source of truth.

Responsibilities:

- Ingest finalized user and assistant turns.
- Ingest validated tool outputs and worker summaries.
- Update semantic, episodic, and commitment stores.
- Resolve salience and decide what is eligible for workspace promotion.
- Preserve provenance pointers.
- Track contradictions, superseded facts, and stale state.

This is the central architecture component, not a background summarizer.

### 3) Workspace Assembler

Deterministic builder for the Executive's next conscious workspace.

Responsibilities:

- Start from active state, not transcript order.
- Retrieve memory shards based on current intent and active entities.
- Apply salience scoring and budget rules.
- Compose the model context in deterministic order.
- Emit observability data for token load and selected memory.

Composition order:

1. Identity and policy invariants.
2. Active thread state.
3. Current user intent.
4. Open commitments and immediate obligations.
5. Relevant entities and project graph state.
6. Retrieved semantic facts.
7. Retrieved episodic shards.
8. Recent turns when needed for local conversational texture.
9. Capability digest and selected tool affordances.
10. Scoped tool schemas only after route selection.

### 4) Router

Cheap route selector for the current turn.

Responsibilities:

- Decide whether the Executive can answer directly.
- Decide whether retrieval, verification, tool use, or a worker is required.
- Select a minimal safe capability pack.
- Keep routing artifacts ephemeral.

The first implementation should be mostly deterministic. An LLM oracle can be
added later for ambiguous capability mapping, but it should not be the default
path.

Router output contract:

```json
{
  "needs_tools": "yes|no",
  "needs_verification": "yes|no",
  "needs_retrieval": "yes|no",
  "complexity": "low|medium|high",
  "route": "direct|retrieve|verify|worker",
  "reason_code": "none|repo_fact|freshness|multi_step|safety|memory_callback",
  "selected_capabilities": ["string"],
  "risk_flags": ["string"]
}
```

Non-negotiable:

- Router artifacts are not added to transcript history.

### 5) Capability Oracle

Optional ephemeral feasibility checker.

Use only when deterministic routing cannot confidently map the request to
available capabilities.

Inputs:

- User intent summary.
- Capability digest.
- Hard runtime constraints.

Outputs:

- feasible: yes | no | partial.
- required_capabilities: string[].
- suggested_route: direct | retrieve | verify | worker.
- risk_flags: enum[].
- confidence: 0..1.

Lifetime:

- One request, then despawn.

### 6) Worker Minds

Ephemeral specialists for bounded tasks requiring deeper tooling, retrieval, or
synthesis.

Responsibilities:

- Execute task packets with explicit objective and done criteria.
- Receive scoped context packs, not broad conversation history.
- Receive scoped tool schemas only for required capabilities.
- Return structured artifacts, not hidden reasoning logs.

Inputs:

- objective.
- done criteria.
- scoped context pack.
- selected tool schemas.
- constraints and budget.

Outputs:

- result_summary.
- evidence_refs.
- confidence.
- open_risks.
- suggested_user_response.
- memory_candidates.

Constraints:

- Max one hop by default.
- Time-boxed.
- No recursive worker trees unless explicitly enabled for an experiment.
- Worker output is an artifact with TTL and promotion rules, not a permanent
  transcript stream.

### 7) Reflective Critic

Optional tiny pre-response quality gate.

Purpose:

- Check workspace integrity before the final response.
- Catch missing evidence, contradictions, or clarification needs.

Output only:

- missing_evidence: yes | no.
- contradiction_detected: yes | no.
- clarify_needed: yes | no.
- action: respond | ask | verify.

Budget:

- Sub-120 tokens.

Non-goal:

- Do not run a generic essay-quality judge.

## Salience Model

Every memory item should have a reason to remain active.

Salience increases with:

- Current user intent.
- Explicit commitment.
- Repeated reference.
- Active project relevance.
- Recent contradiction or correction.
- Unresolved decision.
- Tool/task dependency.
- User-stated preference or design principle.

Salience decays when:

- The related project becomes inactive.
- A commitment is completed or dropped.
- Newer evidence supersedes old state.
- The item has not been activated for a long time.
- It is only conversational texture with no durable value.

Rule:

- Salience controls workspace promotion. It should not delete provenance.

## Per-Turn Flow

1. User turn arrives.
2. Steward extracts current-turn cues: intent, entities, possible commitments,
   memory callbacks, and risk markers.
3. Router selects direct, retrieve, verify, or worker path.
4. Workspace Assembler builds a compact state-first context.
5. Executive answers directly or triggers selected tools/workers.
6. Worker results return as structured artifacts when used.
7. Executive integrates evidence into a final response.
8. Optional Critic checks workspace integrity.
9. Final response is sent.
10. Steward updates semantic, episodic, and commitment stores from finalized
    artifacts.

## 32k Budget Envelope

Steady-state targets:

- Workspace base, active state, commitments, and selected memory: 10k-16k.
- Reserve for response and tool round-trips: 8k-12k.
- Deliberation burst headroom: 4k-8k.
- Safety reserve to avoid recurrent compaction: at least 4k.

Rules:

- Never run near the hard limit by default.
- Extra context is a bonus for tool use and evidence, not a dumping ground.
- Compaction is exceptional recovery, not routine turn behavior.
- If routine turns require compaction, workspace assembly is failing.

## Prompt Contracts

### Executive prompt contract

Includes:

- system_policy.
- active_thread_state.
- current_user_intent.
- commitments.
- relevant_entities.
- retrieved_semantic_facts.
- retrieved_episodic_shards.
- recent_turn_texture when needed.
- capability_digest.
- selected_tool_affordances.

Excludes:

- full_tool_catalog.
- historical_bulk.
- verbose tool archives.
- worker_internal_logs.
- router scratch state.

### Worker prompt contract

Includes:

- scoped_task_packet.
- done criteria.
- scoped_context_pack.
- selected_tool_schemas.
- constraints.

Excludes:

- unrelated tools.
- broad session history.
- unrelated memories.
- Executive private scratch state.

### Steward update contract

Ingests:

- finalized user turn.
- finalized assistant response.
- validated tool outputs.
- worker summaries.
- explicit user corrections.

Writes:

- semantic facts.
- entity and relationship updates.
- episodic event records.
- commitment deltas.
- contradiction records.
- provenance refs.

## Implementation Target In OpenClaw

The first practical target is a personal context engine, not a parallel agent
runtime.

Map this architecture onto the existing context-engine lifecycle:

- bootstrap: import existing transcript into structured stores.
- assemble: build the conscious workspace.
- afterTurn: update semantic, episodic, and commitment stores.
- maintain: perform background cleanup, salience decay, and transcript
  maintenance.
- compact: exceptional recovery path only.

Experimental priority:

1. Build the Steward stores and Workspace Assembler.
2. Add observability for token load, selected memory, and compaction count.
3. Add deterministic routing and scoped capability packs.
4. Add worker tasks for bounded deep synthesis.
5. Add optional Oracle/Critic calls only after metrics justify them.

## Success Criteria

A successful implementation should show:

- Sustained multi-turn coherence at 32k target settings.
- Executive context remains state-first rather than transcript-first.
- Significant reduction in per-turn context load.
- Compaction no longer triggers on most routine turns.
- Commitments survive long conversations.
- Long-horizon callbacks retrieve the right prior state.
- Tool-heavy tasks complete without collapsing conversational continuity.
- Worker outputs do not accumulate as hidden transcript bulk.

## Failure Signals

Architecture is failing if:

- The workspace drifts back toward full-schema/full-history behavior.
- Compaction triggers every few turns during normal usage.
- Retrieval repeatedly misses explicit commitments or major project facts.
- The Executive rediscovers unchanged facts every turn.
- Worker delegation loops.
- Worker summaries become another unbounded memory stream.
- Contradictions are silently overwritten.
- Recent wording wins over newer structured state.

## Experimental Validation Plan

Use local Gemma through LM Studio and/or Ollama.

Scenarios:

1. 20-turn mixed conversation: planning, coding, corrections, follow-ups.
2. Tool-heavy 10-turn sequence: repo reads, command outputs, synthesis.
3. Long-horizon callback test: user refers to early commitments after topic
   drift.
4. Contradiction test: user corrects a previously stored fact.
5. Salience decay test: unrelated old details should not remain in workspace.
6. Worker containment test: worker outputs should help the answer without
   becoming permanent transcript bulk.

Capture:

- per-turn estimated token load.
- workspace section token load.
- selected memory item count.
- memory retrieval hit/miss notes.
- compaction trigger count.
- median and p95 latency.
- commitment recall accuracy.
- contradiction handling accuracy.
- user-visible coherence score.

## Implementation Intent

This architecture is intentionally not industry-default transcript stuffing.

It is a mind-model approach: bounded conscious workspace, structured durable
memory, salience-driven retrieval, scoped procedural affordances, and ephemeral
specialized thought threads.

The key test is:

After 100 turns, can the Executive enter turn 101 with a compact workspace that
contains the right living state, not the accumulated corpse of the conversation?
