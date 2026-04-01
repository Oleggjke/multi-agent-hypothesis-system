import pandas as pd
from state import AgentState

def orchestrator(state: AgentState) -> AgentState:
    df = state["dataset"]

    # собираем описание датасета
    description_parts = []
    description_parts.append(f"Датасет содержит {len(df)} строк и {len(df.columns)} столбцов.")
    description_parts.append(f"Столбцы: {', '.join(df.columns.tolist())}")

    # типы столбцов
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    if numeric_cols:
        description_parts.append(f"Числовые столбцы: {', '.join(numeric_cols)}")
    if categorical_cols:
        description_parts.append(f"Категориальные столбцы: {', '.join(categorical_cols)}")

    # базовая статистика по числовым столбцам
    if numeric_cols:
        stats_str = df[numeric_cols].describe().round(2).to_string()
        description_parts.append(f"Статистика:\n{stats_str}")

    # пропущенные значения
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        missing_str = ", ".join([f"{col}: {cnt}" for col, cnt in missing.items()])
        description_parts.append(f"Пропущенные значения: {missing_str}")
    else:
        description_parts.append("Пропущенных значений нет.")

    dataset_description = "\n".join(description_parts)

    return {
        **state,
        "dataset_description": dataset_description,
        "iteration_count": 0,
        "final_report": []
    }
