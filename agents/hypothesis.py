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
    temperature=0.7
)

def hypothesis_agent(state: AgentState) -> AgentState:
    description = state["dataset_description"]

    messages = [
        SystemMessage(content="""Ты аналитик данных. Твоя задача — формулировать конкретные 
        исследовательские гипотезы на основе описания датасета.
        
        Правила:
        1. Формулируй гипотезы конкретно — указывай названия столбцов
        2. Каждая гипотеза должна быть проверяемой статистически
        3. Верни ровно 5 гипотез в виде JSON списка
        
        Пример формата ответа:
        ["гипотеза 1", "гипотеза 2", "гипотеза 3", "гипотеза 4", "гипотеза 5"]
        """),
        HumanMessage(content=f"""Вот описание датасета:
        
        {description}
        
        Сформулируй 5 исследовательских гипотез. Верни только JSON список, без пояснений.""")
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
        hypotheses = json.loads(content)
    except json.JSONDecodeError:
        # если не распарсилось — делаем список из текста
        hypotheses = [line.strip() for line in content.split("\n") if line.strip()]

    return {
        **state,
        "hypotheses": hypotheses,
        "current_hypothesis": hypotheses[0],
        "current_hypothesis_index": 0
    }
