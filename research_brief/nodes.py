import json
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from research_brief.state import ResearchState
from research_brief.tools import CONTEXT_DIR, RESEARCH_TOOLS

load_dotenv()

planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
research_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1).bind_tools(RESEARCH_TOOLS)
analyst_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
reviewer_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

tool_node = ToolNode(RESEARCH_TOOLS)


def parse_json(text: str) -> dict:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = match.group(1) if match else text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def load_context(state: ResearchState) -> dict:
    print("[load_context] reading project brief")
    path = CONTEXT_DIR / "project_brief.md"
    ctx = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "local_context": ctx,
        "phase": "planning",
        "research_round": 0,
        "revision_round": 0,
        "tool_steps": 0,
        "web_findings": [],
        "knowledge_gaps": [],
        "review_score": 0,
        "error": None,
    }


def planner_node(state: ResearchState) -> dict:
    print(f"[planner] topic: {state['topic']}")
    prompt = f"""You plan research for a product/engineering team.

Topic: {state['topic']}

Project context:
{state.get('local_context', '')[:2500]}

Return 3-4 research questions that need web search.
JSON only:
{{"research_questions": ["...", "..."]}}"""

    resp = planner_llm.invoke([HumanMessage(content=prompt)])
    data = parse_json(resp.content)
    questions = data.get("research_questions", [])[:4]
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

    return {
        "research_questions": questions,
        "phase": "researching",
        "messages": [
            SystemMessage(content=(
                "Research agent. Use web_search, read_local_file, save_to_memory, recall_from_memory. "
                "After 2-3 searches you have enough — stop calling tools."
            )),
            HumanMessage(content=(
                f"Topic: {state['topic']}\n"
                f"Questions:\n" + "\n".join(f"- {q}" for q in questions) + "\n"
                "Recall memory, search web, save important notes."
            )),
        ],
    }


def research_agent_node(state: ResearchState) -> dict:
    from_analyst = bool(state.get("knowledge_gaps"))
    tool_steps = (0 if from_analyst else state.get("tool_steps", 0)) + 1
    print(f"[research] tool step {tool_steps}")

    extra = []
    if from_analyst:
        extra.append(HumanMessage(content=(
            "Fill these gaps:\n" + "\n".join(f"- {g}" for g in state["knowledge_gaps"])
        )))

    resp = research_llm.invoke(state["messages"] + extra)
    out = {"messages": extra + [resp], "tool_steps": tool_steps}
    if from_analyst:
        out["research_round"] = state.get("research_round", 0) + 1
    return out


def research_tools_node(state: ResearchState) -> dict:
    print("[research] running tools")
    result = tool_node.invoke(state)

    findings = []
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage) and "Search results for:" in msg.content:
            findings.append({
                "source": "web_search",
                "round": state.get("research_round", 0),
                "content": msg.content[:3000],
            })

    out = {"messages": result["messages"]}
    if findings:
        out["web_findings"] = findings
    return out


def should_continue_research(state: ResearchState) -> Literal["research_tools", "analyst"]:
    limit = state.get("max_tool_steps", 6)
    if state.get("tool_steps", 0) >= limit:
        print(f"[route] tool step limit ({limit}), -> analyst")
        return "analyst"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "research_tools"
    return "analyst"


def analyst_node(state: ResearchState) -> dict:
    print("[analyst] summarizing")

    findings = "\n\n".join(f.get("content", "")[:1500] for f in state.get("web_findings", []))
    if not findings:
        findings = "(no structured web findings)"

    prompt = f"""Analyze research for topic: {state['topic']}

Questions: {json.dumps(state.get('research_questions', []), ensure_ascii=False)}

Findings:
{findings[:5000]}

JSON only:
{{
  "analysis_summary": "bullet summary",
  "knowledge_gaps": ["gap"] or [],
  "research_complete": true/false
}}"""

    resp = analyst_llm.invoke([HumanMessage(content=prompt)])
    data = parse_json(resp.content)
    gaps = data.get("knowledge_gaps", [])
    complete = data.get("research_complete", len(gaps) == 0)
    if not state.get("web_findings") and not complete:
        complete = True
        gaps = []
        print("[analyst] no web findings, proceeding anyway")
    print(f"[analyst] gaps={len(gaps)}, complete={complete}")

    return {
        "analysis_summary": data.get("analysis_summary", ""),
        "knowledge_gaps": gaps if not complete else [],
        "phase": "writing" if complete else "researching",
    }


def route_after_analysis(state: ResearchState) -> Literal["research_agent", "writer"]:
    gaps = state.get("knowledge_gaps", [])
    n = state.get("research_round", 0)
    limit = state.get("max_research_rounds", 2)

    if gaps and n < limit:
        print(f"[route] back to research ({n}/{limit})")
        return "research_agent"
    print("[route] -> writer")
    return "writer"


def writer_node(state: ResearchState) -> dict:
    rev = state.get("revision_round", 0)
    print(f"[writer] revision {rev}")

    feedback = ""
    if state.get("review_feedback"):
        feedback = f"\nReviewer feedback:\n{state['review_feedback']}"

    prompt = f"""Write a markdown research brief.

Topic: {state['topic']}
Context: {state.get('local_context', '')[:2000]}
Analysis: {state.get('analysis_summary', '')}
Evidence: {json.dumps(state.get('web_findings', [])[:5], ensure_ascii=False)[:3000]}
{feedback}

Sections: Executive Summary, Key Findings, Risks and Open Questions, Recommendations.
600-900 words."""

    resp = writer_llm.invoke([HumanMessage(content=prompt)])
    return {"report_draft": resp.content, "phase": "reviewing"}


def reviewer_node(state: ResearchState) -> dict:
    n = state.get("revision_round", 0) + 1
    print(f"[reviewer] pass {n}")

    prompt = f"""Review this brief (score 1-10, approve if >= 7).

Topic: {state['topic']}
Draft: {state.get('report_draft', '')[:6000]}

JSON only:
{{"review_score": N, "approved": true/false, "review_feedback": "..."}}"""

    resp = reviewer_llm.invoke([HumanMessage(content=prompt)])
    data = parse_json(resp.content)
    score = int(data.get("review_score", 5))
    approved = bool(data.get("approved", score >= 7))
    print(f"[reviewer] score={score}, approved={approved}")

    return {
        "review_score": score,
        "review_feedback": data.get("review_feedback", ""),
        "revision_round": n,
        "phase": "done" if approved else "writing",
    }


def route_after_review(state: ResearchState) -> Literal["writer", "save_report"]:
    score = state.get("review_score", 0)
    n = state.get("revision_round", 0)
    limit = state.get("max_revision_rounds", 2)

    if score >= 7:
        return "save_report"
    if n < limit:
        print(f"[route] back to writer ({n}/{limit})")
        return "writer"
    print("[route] max revisions, saving anyway")
    return "save_report"


def save_report_node(state: ResearchState) -> dict:
    from research_brief.tools import write_report

    slug = re.sub(r"[^\w\s-]", "", state["topic"].lower())
    slug = re.sub(r"[-\s]+", "_", slug).strip("_")[:50]
    filename = f"brief_{slug}.md"

    body = (
        f"# Research Brief: {state['topic']}\n\n"
        f"Score: {state.get('review_score')}/10 | "
        f"research rounds: {state.get('research_round')} | "
        f"revision rounds: {state.get('revision_round')}\n\n"
        f"{state.get('report_draft', '')}"
    )
    path = write_report.invoke({"filename": filename, "content": body})
    print(f"[save] {path}")

    return {"output_path": f"data/reports/{filename}", "phase": "done"}
