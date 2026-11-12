from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from research_brief.nodes import (
    analyst_node,
    load_context,
    planner_node,
    research_agent_node,
    research_tools_node,
    reviewer_node,
    route_after_analysis,
    route_after_review,
    save_report_node,
    should_continue_research,
    writer_node,
)
from research_brief.state import ResearchState


def build_graph(checkpoint: bool = True):
    g = StateGraph(ResearchState)

    g.add_node("load_context", load_context)
    g.add_node("planner", planner_node)
    g.add_node("research_agent", research_agent_node)
    g.add_node("research_tools", research_tools_node)
    g.add_node("analyst", analyst_node)
    g.add_node("writer", writer_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("save_report", save_report_node)

    g.set_entry_point("load_context")
    g.add_edge("load_context", "planner")
    g.add_edge("planner", "research_agent")

    g.add_conditional_edges(
        "research_agent",
        should_continue_research,
        {"research_tools": "research_tools", "analyst": "analyst"},
    )
    g.add_edge("research_tools", "research_agent")

    g.add_conditional_edges(
        "analyst",
        route_after_analysis,
        {"research_agent": "research_agent", "writer": "writer"},
    )

    g.add_edge("writer", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"writer": "writer", "save_report": "save_report"},
    )
    g.add_edge("save_report", END)

    checkpointer = MemorySaver() if checkpoint else None
    return g.compile(checkpointer=checkpointer)
