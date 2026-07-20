# Ideas & Experiments

### [Recursive Observer Framework](ro_framework/ro_framework.md)
* Theoretical framework that models AI systems as observers — configurations defined by four structural properties: a boundary (which inputs are external vs internal), a mapping (how external signals become internal representations), a resolution (finite distinguishability per dimension), and memory (correlation of internal states across time). Every input and output dimension is typed as a Degree of Freedom (DoF) — polar, scalar, categorical, or derived — giving the observer a structured, typed interface to its environment.
* It defines knowledge not as binary correctness but as a four-dimensional profile: correlation between internal state and the external signal, systematic bias, random noise, and calibration (whether stated confidence matches actual accuracy). This yields graded knowledge types — strong, weak, false, or uncertain — rather than a single accuracy number. Consciousness is defined structurally as recursive self-modeling: an observer whose internal-to-internal mapping (self-model) has the same architectural type as its external-to-internal mapping (world model). The quality of this self-modeling is measurable — self-prediction accuracy, calibration error, metacognitive consistency, limitation awareness — without making philosophical claims about phenomenal experience.
### [Recursive Observer Framework - Python Lib](https://github.com/IdentityOverflow/ROFramework-PyLib)
* A Python library implemented based on the theoretical framework above. Wrapping any model as an Observer and asking structured questions about what it knows, how well-calibrated it is, and whether it can model itself.
* Most ML tools focus on training models. This library focuses on understanding them after the fact.
  
Graded knowledge assessment — Go beyond accuracy. When you wrap a model and feed it data, the library tracks paired (input, output) history and computes a four-dimensional knowledge profile:
* Is the model's internal state correlated with the input? (not just "right or wrong")
* Is there systematic bias? (consistently wrong in one direction)
* How noisy is the mapping? (inconsistent outputs for similar inputs)
* Is uncertainty calibrated? (when it says "80% confident", is it right 80% of the time?)
### [Cognitive Workspace Architecture](cognitive_workspace_architecture.md)
* Context is not memory: a context window is a conscious workspace — a narrow, active field where goals, entities, commitments, retrieved facts, and possible actions are bound together — while durable memory lives outside it as structured state, indexed episodes, and provenance-backed artifacts. Defines the stores, components, and per-turn contracts (the *anatomy*) that [Ghost in the Proxy](https://github.com/IdentityOverflow/ghost-in-the-proxy) implements; the Cognitive Runtime below supplies the dynamics (*physiology*), and OCA the developmental arc.

### [Cognitive Runtime Architecture](cognitive_runtime_with_SLMs.md)
The Cognitive Runtime is a system architecture that enables a **small language model** to maintain coherent, long-running conversations by simulating **multi-thread cognition**, memory consolidation, and predictive reasoning.

The runtime treats the LLM as a **stateless reasoning engine** while managing memory, attention, and interaction pacing externally.

Primary goals:

* transcend LLM context window limitations
* maintain conversation coherence indefinitely
* simulate multi-thread cognition
* support reflection and background processing
* provide natural, paced interaction with the user

**Implemented**: the thread-activation dynamics, decay, question attractors, and cued reinjection from this document are now running code — see [Ghost in the Proxy](https://github.com/IdentityOverflow/ghost-in-the-proxy) below.
### [Organic Cognitive Architecture (OCA)](organic_cognitive_architecture_oca.md)
* A reservoir-based, continuously learning cognitive architecture inspired by biological brains but grounded in control theory and reinforcement learning.
* Experiment idea: Tuning fork principle - Use EEG signals (e.g., from Neurosity Crown) to extract dominant brainwave frequencies (delta, theta, alpha, beta, gamma), then tune a reservoir computing system so its internal dynamics resonate at matching frequencies. Look for higher-coherence dynamics.
### [Dynamic System Prompt Framework - Prototype 1](https://github.com/IdentityOverflow/DynamicSystemPrompt-Prototype)
* A modular framework for building dynamic prompts with pluggable components for AI systems.
### [Modular Dynamic Context System (MDCS) - Prototype 2](https://github.com/IdentityOverflow/MDCS)
* A full-stack web application for managing AI conversations with dynamic, modular system prompts. The system allows system prompts to be composed from reusable modules that can execute Python scripts, call AI models, and maintain state across conversations.
### [Ghost in the Proxy](https://github.com/IdentityOverflow/ghost-in-the-proxy)
* A persistent, structured, living mind for any OpenAI-compatible model — instead of a context window full of dead transcript. An OpenAI-compatible proxy whose middleware treats the client transcript as a sensory event stream and assembles the model's context fresh every request: structured ledger (facts / decisions with status / commitments with triggers), CRS thread-activation dynamics with cued recall, a verbatim `recall` tool over the raw event store, per-turn tool routing, and tool-payload containment. Synthesizes the Cognitive Runtime Architecture (above) with the [Cognitive Workspace Architecture](cognitive_workspace_architecture.md) anatomy, gated phase by phase on an eval suite.
* Founding result: gemma-4-12B at an 8k window went from 57% (transcript-stuffing baseline) to 86% with the structured ledger, and to 31/32 with the full architecture — flat token curves, and survival at 4k where the baseline aborts mid-conversation.
### [LLM Passthrough Endpoint](https://github.com/IdentityOverflow/LLM-passthrough-endpoint)
* A faithful OpenAI-compatible passthrough proxy (the shell Ghost in the Proxy lives in): multi-provider routing, model aliasing, SSE streaming, and a pinned fidelity contract — unmodeled fields pass through, tool round-trips intact, backend errors mirrored. Use it as-is, or as a clean base for your own inference-time scaffolding.
