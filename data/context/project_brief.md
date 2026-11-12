# Project Brief: LangGraph Workshop — Curriculum Draft

**Team:** AI Platform Guild, internal engineering enablement  
**Author:** Platform Learning  
**Status:** Draft v0.3  
**Date:** June 2026

## Context

We are putting together a hands-on workshop on **LangGraph** for engineers who already know basic LangChain but have not built multi-agent workflows.

Format: two 90-minute sessions (concepts + live coding). Audience: backend and ML engineers, ~25 people, mixed experience with LLM apps.

## What the workshop must demonstrate

- Multi-agent setup with distinct roles — not a single monolithic prompt.
- **Cyclic graphs**: at least one loop where execution returns to an earlier step.
- **Tools** used by agents (web search, file I/O, simple persistent memory).
- **Shared state** passed between nodes and used for routing decisions.
- A concrete artifact at the end (e.g. a generated report file students can open).

Live demo budget: ~5–8 minutes per run; total hands-on walkthrough ~45 minutes per session.

## Evaluation criteria for demo outputs

1. **Relevance** — addresses the stated research topic directly.
2. **Evidence** — uses retrieved data, not only model prior knowledge.
3. **Context fit** — ties findings back to this brief where it makes sense.
4. **Structure** — clear sections: summary, findings, risks/open questions, recommendations.
5. **Actionability** — recommendations a team could actually use when designing the workshop labs.

## Open design questions

- How many agent roles is enough for teaching without overload?
- Where to put loop guards so live demos do not run forever?
- Which model to use per role (cost vs. quality tradeoff)?

## Out of scope

Production deployment, auth, billing, and full observability stack — mention in passing only.
