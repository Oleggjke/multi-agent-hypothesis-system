import pandas as pd
from graph import build_graph


def main():
    # загружаем датасет
    print("Загрузка датасета...")
    df = pd.read_csv("data/dataset.csv")
    print(f"Загружено {len(df)} строк, {len(df.columns)} столбцов")

    # начальное состояние
    initial_state = {
        "dataset": df,
        "dataset_description": None,
        "hypotheses": None,
        "current_hypothesis": None,
        "current_hypothesis_index": 0,
        "selected_method": None,
        "method_reasoning": None,
        "analysis_result": {},
        "verification_passed": False,
        "verification_feedback": "",
        "iteration_count": 0,
        "interpretation": None,
        "final_report": []
    }

    # собираем и запускаем граф
    print("Запуск мультиагентной системы...\n")
    graph = build_graph()
    final_state = graph.invoke(initial_state)

    # выводим отчёт
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60 + "\n")

    for entry in final_state["final_report"]:
        print(entry)

    # сохраняем отчёт в файл
    with open("output/report.txt", "w", encoding="utf-8") as f:
        f.write("ИТОГОВЫЙ ОТЧЁТ\n")
        f.write("=" * 60 + "\n\n")
        for entry in final_state["final_report"]:
            f.write(entry + "\n")

    print("\nОтчёт сохранён в output/report.txt")


if __name__ == "__main__":
    main()
