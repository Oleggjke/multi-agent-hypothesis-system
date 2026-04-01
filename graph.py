from langgraph.graph import StateGraph, END
from state import AgentState
from agents.orchestrator import orchestrator
from agents.hypothesis import hypothesis_agent
from agents.method_selector import method_selector_agent
from agents.analyst import analyst_agent
from agents.verificator import verificator_agent
from agents.interpreter import interpreter_agent


def should_retry(state: AgentState) -> str:
    # если верификация не прошла — возвращаемся к выбору метода
    if not state["verification_passed"]:
        return "retry"
    return "continue"


def should_continue(state: AgentState) -> str:
    # если ещё есть гипотезы — продолжаем
    current_index = state["current_hypothesis_index"]
    hypotheses = state["hypotheses"]
    if current_index < len(hypotheses):
        return "next"
    return "done"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # добавляем узлы
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("hypothesis_agent", hypothesis_agent)
    graph.add_node("method_selector", method_selector_agent)
    graph.add_node("analyst", analyst_agent)
    graph.add_node("verificator", verificator_agent)
    graph.add_node("interpreter", interpreter_agent)

    # задаём точку входа
    graph.set_entry_point("orchestrator")

    # фиксированные переходы
    graph.add_edge("orchestrator", "hypothesis_agent")
    graph.add_edge("hypothesis_agent", "method_selector")
    graph.add_edge("method_selector", "analyst")
    graph.add_edge("analyst", "verificator")

    # условный переход после верификации
    graph.add_conditional_edges(
        "verificator",
        should_retry,
        {
            "retry": "method_selector",
            "continue": "interpreter"
        }
    )

    # условный переход после интерпретации
    graph.add_conditional_edges(
        "interpreter",
        should_continue,
        {
            "next": "method_selector",
            "done": END
        }
    )

    return graph.compile()
