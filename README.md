# Idea drawer

### [Recursive Observer Framework](ro_framework/ro_framework.md)
* Philosophical thesis aimed at bridging the language gap between disparate domains, with the goal of being formalizable and translated into code for use in AI implementation.
### [Recursive Observer Framework - Python Lib](https://github.com/IdentityOverflow/ROFramework-PyLib)
* A Python library implemented based on the theoretical framework above. Wrapping any model as an Observer and asking structured questions about what it knows, how well-calibrated it is, and whether it can model itself.
* Most ML tools focus on training models. This library focuses on understanding them after the fact.
  
Graded knowledge assessment — Go beyond accuracy. When you wrap a model and feed it data, the library tracks paired (input, output) history and computes a four-dimensional knowledge profile:
* Is the model's internal state correlated with the input? (not just "right or wrong")
* Is there systematic bias? (consistently wrong in one direction)
* How noisy is the mapping? (inconsistent outputs for similar inputs)
* Is uncertainty calibrated? (when it says "80% confident", is it right 80% of the time?)
### [Organic Cognitive Architecture (OCA)](organic_cognitive_architecture_oca.md)
* A reservoir-based, continuously learning cognitive architecture inspired by biological brains but grounded in control theory and reinforcement learning.
### [Dynamic System Prompt Framework - Prototype 1](https://github.com/IdentityOverflow/DynamicSystemPrompt-Prototype)
* A modular framework for building dynamic prompts with pluggable components for AI systems.
### [Modular Dynamic Context System (MDCS) - Prototype 2](https://github.com/IdentityOverflow/MDCS)
* A full-stack web application for managing AI conversations with dynamic, modular system prompts. The system allows system prompts to be composed from reusable modules that can execute Python scripts, call AI models, and maintain state across conversations.
### [Experimental base](https://github.com/IdentityOverflow/LLM-experimental-base)
* An OpenAI API inference passthrough, where you can add additional scaffolding and capabilities to the model output before passing it to your application. LangChain and LangGraph libraries ready to go.
