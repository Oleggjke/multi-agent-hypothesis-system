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
    temperature=0
)

def verificator_agent(state: AgentState) -> AgentState:
    hypothesis = state["current_hypothesis"]
    method = state["selected_method"]
    reasoning = state["method_reasoning"]
    result = state["analysis_result"]
    iteration = state.get("iteration_count", 0)

    messages = [
        SystemMessage(content="""Ты агент верификации результатов анализа данных.
        Твоя задача — проверить корректность результатов и соответствие метода гипотезе.
        
        Проверяй следующее:
        1. Соответствует ли выбранный метод гипотезе
        2. Есть ли ошибки в результатах (поле error)
        3. Являются ли результаты статистически осмысленными
        4. Достаточно ли данных для вывода
        
        Верни ответ строго в формате JSON:
        {
            "passed": true или false,
            "feedback": "объяснение если не прошло, или ok если прошло"
        }
        """),
        HumanMessage(content=f"""Гипотеза: {hypothesis}
        
        Выбранный метод: {method}
        Обоснование выбора: {reasoning}
        
        Результаты анализа: {json.dumps(result, ensure_ascii=False)}
        
        Верификация пройдена? Верни JSON.""")
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    # убираем markdown если модель его добавила
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        verification = json.loads(content)
        passed = verification.get("passed", False)
        feedback = verification.get("feedback", "")
    except json.JSONDecodeError:
        passed = False
        feedback = "не удалось распарсить ответ верификатора"

    # если слишком много итераций — принудительно пропускаем
    if iteration >= 2:
        passed = True
        feedback = "принудительный пропуск после 2 итераций"

    return {
        **state,
        "verification_passed": passed,
        "verification_feedback": feedback,
        "iteration_count": iteration + 1
    }
