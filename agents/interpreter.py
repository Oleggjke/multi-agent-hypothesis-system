from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.3
)

def interpreter_agent(state: AgentState) -> AgentState:
    hypothesis = state["current_hypothesis"]
    method = state["selected_method"]
    result = state["analysis_result"]
    description = state["dataset_description"]

    messages = [
        SystemMessage(content="""Ты аналитик данных. Твоя задача — интерпретировать 
        результаты статистического анализа на понятном русском языке.
        
        Правила:
        1. Объясни что означают числовые результаты
        2. Чётко скажи подтверждена гипотеза или опровергнута
        3. Укажи уровень уверенности в выводе
        4. Пиши кратко и по делу, без лишних слов
        5. Не используй markdown разметку
        """),
        HumanMessage(content=f"""Датасет: {description}
        
        Гипотеза: {hypothesis}
        
        Метод анализа: {method}
        
        Результаты: {json.dumps(result, ensure_ascii=False)}
        
        Напиши интерпретацию результатов.""")
    ]

    response = llm.invoke(messages)
    interpretation = response.content.strip()

    report_entry = f"""Гипотеза: {hypothesis}
Метод: {method}
Результаты: {json.dumps(result, ensure_ascii=False, indent=2)}
Интерпретация: {interpretation}
{"=" * 60}"""

    current_report = state.get("final_report", [])

    hypotheses = state["hypotheses"]
    current_index = state["current_hypothesis_index"]
    next_index = current_index + 1

    if next_index < len(hypotheses):
        next_hypothesis = hypotheses[next_index]
    else:
        next_hypothesis = None

    return {
        **state,
        "interpretation": interpretation,
        "final_report": current_report + [report_entry],
        "current_hypothesis": next_hypothesis,
        "current_hypothesis_index": next_index,
        "selected_method": None,
        "method_reasoning": None,
        "analysis_result": {"params": {}},
        "verification_passed": False,
        "verification_feedback": "",
        "iteration_count": 0
    }
