# Cognitive Runtime Specification (CRS)

### Version 1.0

## Purpose

The Cognitive Runtime is a system architecture that enables a **small language model** to maintain coherent, long-running conversations by simulating **multi-thread cognition**, memory consolidation, and predictive reasoning.

The runtime treats the LLM as a **stateless reasoning engine** while managing memory, attention, and interaction pacing externally.

Primary goals:

* transcend LLM context window limitations
* maintain conversation coherence indefinitely
* simulate multi-thread cognition
* support reflection and background processing
* provide natural, paced interaction with the user

---

# 1. Core Principles

### LLM as Reasoning Engine

The LLM is responsible only for:

* reasoning
* summarization
* semantic updates
* generating responses

The runtime manages:

* attention (thread activation)
* memory organization
* thread lifecycle
* scheduling
* interaction pacing

---

### Multi-Thread Cognition

Conversation state is represented as **threads** rather than raw message history.

Threads represent:

* topics
* questions
* predictions
* cognitive processes

Working memory contains only a **small subset of threads**.

Recommended working memory size:

```text
3–4 active topic threads
1 narrative thread
```

---

### Memory Layers

The system contains three memory layers.

```
Working Memory
    Active threads used in reasoning

Thread Memory
    Warm threads with moderate activation

Archive Memory
    Archived threads stored in database/vector store
```

---

# 2. Thread Model

Each thread represents a cognitive unit.

## Thread Schema

```json
Thread {
  "id": string,
  "topic": string,

  "role": string,
  "focus": string,

  "summary": string,
  "anchors": [string],

  "open_questions": [string],
  "hypotheses": [string],

  "predictions": [],

  "activation": float,
  "importance": float,
  "confidence": float,

  "created_step": int,
  "last_updated_step": int
}
```

---

## Thread Categories

### Topic Threads

Represent subjects or ongoing discussions.

Examples:

```
LLM cognitive architecture
thread activation scoring
LangGraph orchestration
```

---

### Functional Threads

Represent cognitive mechanisms.

Examples:

```
Memory Watcher
Prediction Engine
Thread Consolidator
Concept Decomposer
```

These are persistent singletons.

---

### Inquiry Threads

Maintain open questions or hypotheses.

Example:

```
Should thread importance decay over time?
```

Questions act as **attention attractors**.

---

### Narrative Thread

Maintains system identity and conversation trajectory.

Example narrative:

```
The system is exploring architectures for long-term LLM cognition
using dynamic threads and memory consolidation.
```

---

# 3. Thread Activation Dynamics

Thread attention is governed by a **mathematical activation model**.

Activation value:

```
activation ∈ [0,1]
```

---

## Activation Update

Each cycle:

```
A(t+1) = decay * A(t) + input + reinforcement
```

Recommended parameters:

```
decay = 0.85
```

---

### Input Term

Measures relevance of current message to thread.

```
input = similarity(user_message, thread_embedding) * relevance_weight
```

Recommended:

```
relevance_weight ≈ 0.6
```

---

### Reinforcement Term

Occurs when thread contributes to reasoning.

```
reinforcement = importance * reinforcement_weight
```

Recommended:

```
reinforcement_weight ≈ 0.2
```

---

### Saturation

```
activation = min(activation, 1.0)
```

---

### Inertia Constraint

Prevent rapid thread switching:

```
new_activation > current_activation + margin
```

Recommended:

```
margin = 0.1
```

---

# 4. Thread Importance

Importance changes slowly and determines memory stability.

Update rule:

```
importance = importance * 0.98 + 0.02 * activation
```

Importance increases when:

* user revisits topic
* thread contributes to insights
* decisions depend on thread

---

# 5. Thread Lifecycle

Threads move between states:

```
ACTIVE → WARM → ARCHIVED
```

---

## Activation Thresholds

Example thresholds:

```
active_threshold = 0.5
archive_threshold = 0.1
```

---

## Archival Condition

```
activation < 0.1 AND importance < 0.2
```

Archived threads remain searchable.

---

# 6. Memory Reinjection

Archived threads can be reactivated via semantic search.

Process:

```
query_embedding = embed(user_message + active_threads)
```

Vector search returns candidate memories.

Reinjection score:

```
reinjection_score =
    similarity(query, memory)
  * memory_importance
```

Reactivation condition:

```
reinjection_score > 0.7
```

---

# 7. Runtime State Machine

The system operates in three modes.

```
INTERACTION
REFLECTION
IDLE
```

---

## Interaction Mode

Triggered by user input.

Pipeline:

```
User message
↓
activation update
↓
memory watcher
↓
context assembly
↓
LLM reasoning
↓
response output
↓
thread updates
↓
candidate outputs
```

Target runtime:

```
1–3 seconds
```

---

## Reflection Mode

Short idle periods trigger light cognitive processing.

Examples:

```
generate open questions
update predictions
refine summaries
extract memory
```

Only one task runs per cycle.

---

## Idle Mode

Longer idle periods trigger deeper consolidation.

Examples:

```
thread merging
memory compression
narrative update
archive pruning
```

These cycles may take longer.

---

# 8. Context Assembly

Prompt structure:

```
SYSTEM PROMPT

COGNITIVE STATE
    system narrative
    active threads
    conversation summary

USER MESSAGE
```

Thread summaries should remain concise:

```
80–120 tokens
```

---

# 9. LLM Output Format

LLM responses must be structured.

Example:

```json
{
  "response": "...",

  "thread_updates": [
    {
      "thread_id": "...",
      "summary_update": "...",
      "new_questions": []
    }
  ],

  "new_threads": [],

  "candidate_outputs": [],

  "prediction": {
    "topics": [],
    "confidence": 0.0
  }
}
```

The runtime applies these updates deterministically.

---

# 10. Prediction Loop

Prediction thread forecasts user behavior.

Example prediction:

```
predicted_topics = ["implementation", "activation scoring"]
```

When next user message arrives:

```
error = 1 - similarity(predicted_topics, actual_topics)
```

Prediction error influences thread importance.

---

# 11. Question Attractors

Open questions increase activation when relevant.

Rule:

```
if similarity(user_message, question) > threshold:
    activation += 0.25
```

Answered questions become anchors.

---

# 12. Output Scheduler

Not all generated insights are immediately sent.

Candidate outputs enter a queue.

Priority score:

```
priority =
    importance
  + novelty
  + relevance
  - spam_penalty
```

---

## Timing Rules

Example pacing constraints:

```
minimum_gap = 2 seconds
max_followups = 2 per user message
```

Old candidate outputs decay:

```
priority *= decay_rate
```

---

# 13. Interaction Model

The system supports human-like conversation flow.

Example:

```
User message
↓
primary response
↓
possible follow-up insight
↓
optional clarification question
```

Follow-ups must respect pacing rules.

---

# 14. Background Consolidation

During idle periods the system performs maintenance.

Tasks include:

```
thread merging
memory extraction
narrative update
archive pruning
```

Thread merging condition:

```
similarity(threadA, threadB) > 0.85
```

Merged threads unify anchors and summaries.

---

# 15. Global Runtime State

Example runtime state object:

```json
{
  "threads": [],
  "active_threads": [],
  "candidate_outputs": [],
  "conversation_summary": "",
  "system_narrative": "",
  "predictions": [],
  "timers": {},
  "last_user_message": ""
}
```

---

# 16. Scheduler Priority

Task priority order:

```
HIGH
    user interaction

MEDIUM
    thread updates
    prediction evaluation

LOW
    memory consolidation
    thread merging
```

---

# 17. Performance Targets

With an 8B local model:

Typical cycle durations:

```
interaction cycle ≈ 2 seconds
reflection cycle ≈ 5 seconds
idle cycle ≈ 10–30 seconds
```

Total LLM calls should remain limited to maintain responsiveness.

---

# 18. Expected Emergent Behavior

With the above architecture the system should exhibit:

* persistent conversational coherence
* stable topic tracking
* natural follow-up insights
* memory recall over long time spans
* adaptive prediction of user behavior

---

# 19. Implementation Guidance

Key design rule:

The LLM **does not manage memory structures directly**.

The runtime is responsible for:

* thread scoring
* scheduling
* memory storage
* activation dynamics

The LLM only proposes semantic updates.

---

# 20. Summary

The Cognitive Runtime provides:

```
thread-based attention
memory consolidation
prediction feedback
background cognition
interaction pacing
```

This architecture enables a small LLM to function as a **persistent cognitive agent** capable of maintaining coherent conversations indefinitely.

---

If you'd like, I can also produce **one extremely useful addition for implementation**:

A **visual architecture diagram + component map** (scheduler, memory manager, thread manager, LLM interface) so the whole system is easier to implement and reason about.

