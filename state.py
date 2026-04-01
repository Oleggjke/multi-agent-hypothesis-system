from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    # входные данные
    dataset: Optional[object]
    dataset_description: Optional[str]

    # гипотезы
    hypotheses: Optional[List[str]]
    current_hypothesis: Optional[str]
    current_hypothesis_index: Optional[int]

    # метод анализа
    selected_method: Optional[str]
    method_reasoning: Optional[str]

    # результаты вычислений
    analysis_result: Optional[dict]

    # верификация
    verification_passed: Optional[bool]
    verification_feedback: Optional[str]
    iteration_count: Optional[int]

    # интерпретация
    interpretation: Optional[str]

    # итоговый отчёт
    final_report: Optional[List[str]]
