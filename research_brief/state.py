import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

Phase = Literal["planning", "researching", "analyzing", "writing", "reviewing", "done", "failed"]


class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str
    research_questions: list[str]
    local_context: str
    web_findings: Annotated[list[dict], operator.add]
    analysis_summary: str
    knowledge_gaps: list[str]
    report_draft: str
    review_feedback: str
    review_score: int
    research_round: int
    revision_round: int
    tool_steps: int
    max_research_rounds: int
    max_revision_rounds: int
    max_tool_steps: int
    phase: Phase
    output_path: str
    error: str | None
