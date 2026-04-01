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

def method_selector_agent(state: AgentState) -> AgentState:
    hypothesis = state["current_hypothesis"]
    description = state["dataset_description"]
    feedback = state.get("verification_feedback", "")

    feedback_block = ""
    if feedback:
        feedback_block = f"""
        Предыдущий метод не прошёл верификацию.
        Причина: {feedback}
        Выбери другой метод.
        """

    messages = [
        SystemMessage(content="""Ты методолог анализа данных. Твоя задача — выбрать 
        наиболее подходящий метод анализа для проверки гипотезы.
        
        Доступные методы:
        - correlation: корреляционный анализ между двумя числовыми столбцами
        - linear_regression: линейная регрессия для предсказания числового значения
        - logistic_regression: логистическая регрессия для предсказания категории
        - kmeans: кластеризация для группировки объектов
        - ttest: t-тест для сравнения средних двух групп
        - descriptive: описательная статистика для одного столбца
        
        Верни ответ строго в формате JSON:
        {
            "method": "название метода",
            "reasoning": "обоснование выбора",
            "params": {
                "col1": "название столбца",
                "col2": "название столбца"
            }
        }
        
        В params указывай только те поля которые нужны для выбранного метода:
        - correlation: col1, col2
        - linear_regression: target, features (список)
        - logistic_regression: target, features (список)
        - kmeans: features (список), n_clusters (число)
        - ttest: column, group_column (group_column должен быть именем существующего столбца, не выражением вроде 'age < 18')
        - descriptive: column
        """),
        HumanMessage(content=f"""Описание датасета:
        {description}
        
        Гипотеза для проверки:
        {hypothesis}
        
        {feedback_block}
        
        Выбери метод и верни JSON. Без пояснений.""")
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
        result = json.loads(content)
        method = result.get("method", "descriptive")
        reasoning = result.get("reasoning", "")
        params = result.get("params", {})
    except json.JSONDecodeError:
        method = "descriptive"
        reasoning = "не удалось распарсить ответ модели"
        params = {}

    return {
        **state,
        "selected_method": method,
        "method_reasoning": reasoning,
        "analysis_result": {**state.get("analysis_result", {}), "params": params}
    }
