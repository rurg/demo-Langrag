import sys
import uuid

from dotenv import load_dotenv

from research_brief.graph import build_graph

load_dotenv()

DEFAULT_TOPIC = "LangGraph patterns for multi-agent orchestration with tool use"


def run(topic: str = DEFAULT_TOPIC, max_research_rounds: int = 2, max_revision_rounds: int = 2, max_tool_steps: int = 6):
    print(f"\nTopic: {topic}\n")

    graph = build_graph(checkpoint=True)
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 40,
    }

    state = {
        "messages": [],
        "topic": topic,
        "research_questions": [],
        "local_context": "",
        "web_findings": [],
        "analysis_summary": "",
        "knowledge_gaps": [],
        "report_draft": "",
        "review_feedback": "",
        "review_score": 0,
        "research_round": 0,
        "revision_round": 0,
        "tool_steps": 0,
        "max_research_rounds": max_research_rounds,
        "max_revision_rounds": max_revision_rounds,
        "max_tool_steps": max_tool_steps,
        "phase": "planning",
        "output_path": "",
        "error": None,
    }

    result = graph.invoke(state, config=config)

    print("\n--- done ---")
    print(f"phase: {result.get('phase')}")
    print(f"score: {result.get('review_score')}/10")
    print(f"file:  {result.get('output_path')}")
    print(f"research rounds: {result.get('research_round')}, revision rounds: {result.get('revision_round')}")
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--graph":
        print(build_graph(checkpoint=False).get_graph().draw_mermaid())
    else:
        t = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TOPIC
        run(t)
