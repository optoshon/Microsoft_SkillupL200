# Microsoft L200 AI SkillUp Training
**Session Plan by TekFrameworks** 

---

## Confidentiality Notice
This repository is proprietary to **TekFrameworks Consulting Pvt. Ltd.** and is intended solely for sharing the training materials. It may not be shared with third parties. 
 

---

## Session Overview 

| Module | Topic                                                       | Duration (Days) |
| :----- | :---------------------------------------------------------- | :-------------- |
| **M1** | Pretrained Transformers (BERT, GPT, T5)                     | 2               |
| **M2** | Building SLMs from LLMs: Distilling, Shrinking, Fine-tuning | 2               |
| **M3** | Retrieval Augmented Generation (RAG)                        | 3               |
| **M4** | Multi-modal Models, Multi-lingual Models                    | 3               |
| **M5** | Agentic AI: Design to Prototype to Product                  | 4               |
| **M6** | Hackathon: Designing an Intelligent AI System               | 4               |

****
## Module Details

### M1 - Pretrained Transformers (BERT, GPT, T5)
* **Description:** Introduces the internal mechanics of transformer-based language models. You will work with a pretrained model to observe text generation and examine the relationship between model size, memory usage, and performance.
* **Tasks:**
    * Run inference on a pretrained transformer model.
    * Observe how tokenization and embeddings influence outputs.
    * Apply model compression techniques and measure performance impact.
    * Compare behavior before and after optimization.
* **Lab Theme:** Understanding how large language models actually work.
* **Outcome(s):** Understanding that language models are numerical architectures that can be analyzed, measured, and optimized rather than "black boxes." 

### M2 - Building SLMs from LLMs: Distilling, Shrinking, Fine-tuning
* **Description:** Demonstrates how smaller models can be adapted to outperform larger general-purpose models for specific tasks through fine-tuning and parameter-efficient techniques.
* **Tasks:**
    * Fine-tune a compact model for a target task.
    * Apply modern lightweight tuning techniques.
    * Compare performance against a larger untuned model.
    * Study the trade-off between model size, cost, and effectiveness.
* **Lab Theme:** Creating focused small language models for specific tasks.
* **Outcome(s):** Insight into how enterprises build efficient, task-specific models and the trade-off between functionality and efficiency.

### M3 - Retrieval Augmented Generation (RAG)
* **Description:** Focuses on integrating external knowledge sources into language models to produce reliable, context-aware responses. 
* **Tasks:**
    * Build a basic retrieval-based assistant.
    * Experiment with different document chunking strategies.
    * Integrate vector search for context retrieval.
    * Evaluate how retrieval quality impacts answer quality. 
* **Lab Theme:** Solving hallucination by grounding models in knowledge.
* **Outcome(s):** Learning how retrievals work and identifying metrics used to evaluate retrieval systems.

### M4 - Multi-modal Models & Multi-lingual Models 
* **Description:** Introduces multimodal retrieval systems that combine text, structured data, and visual information into a unified AI assistant. 
* **Tasks:**
    * Work with knowledge sources containing text, tables, and images. 
    * Build a system that retrieves relevant information across all modalities.
    * Observe how combining modalities improves response accuracy.
    * Study how multimodal context is constructed for AI models.
* **Lab Theme:** Building systems that understand text, tables, and images together.
* **Outcome(s):** Understanding how modern enterprise AI systems integrate multiple knowledge formats into a single reasoning pipeline.

### M5 - Agentic AI: Design to Prototype to Product
* **Description:** Contrasts linear AI workflows with agentic systems that can plan, draft, evaluate, and improve their own outputs. 
* **Tasks:**
    * Build a traditional AI workflow for a business task.
    * Build an agentic version of the same system. 
    * Observe how planning, evaluation, and iteration improve outputs.
    * Compare the strengths and limitations of both approaches. 
* **Lab Theme:** From simple workflows to autonomous AI agents. 
* **Outcome(s):** Learning when to use workflows versus agents while designing real-world AI applications.

---

## M6 - Hackathon: Designing an Intelligent AI System
**Context:** A team-based challenge presenting a realistic enterprise scenario where an AI system completes operational tasks with a human in the loop.

### The Challenge 
Participants must design and implement an AI-driven system that can:
* Interpret user requests accurately. 
* Decide the appropriate course of action. 
* Use available knowledge sources effectively. 
* Handle incomplete, ambiguous, or inappropriate inputs gracefully.

### Focus & Assessment 
The focus is on designing a structured AI solution for a practical business problem rather than just a chatbot. The challenge assesses the ability to:
* Think in terms of system design rather than isolated prompts.
* Combine different AI techniques (learned in M1-M5) into a cohesive workflow. 
* Apply guardrails and decision logic appropriately. 
* Design prompts and flows that produce reliable outcomes. 
* Communicate system architecture clearly. 