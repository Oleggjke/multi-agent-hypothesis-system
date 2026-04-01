import pandas as pd
from state import AgentState
from tools.correlation import run_correlation
from tools.regression import run_linear_regression, run_logistic_regression
from tools.clustering import run_kmeans
from tools.statistics import run_descriptive, run_ttest

def analyst_agent(state: AgentState) -> AgentState:
    df = state["dataset"]
    method = state["selected_method"]
    params = state.get("analysis_result", {}).get("params", {})

    result = {}

    try:
        if method == "correlation":
            col1 = params.get("col1")
            col2 = params.get("col2")
            if col1 and col2:
                result = run_correlation(df, col1, col2)
            else:
                result = {"error": "не указаны столбцы для корреляции"}

        elif method == "linear_regression":
            target = params.get("target")
            features = params.get("features", [])
            if target and features:
                result = run_linear_regression(df, target, features)
            else:
                result = {"error": "не указаны target или features"}

        elif method == "logistic_regression":
            target = params.get("target")
            features = params.get("features", [])
            if target and features:
                result = run_logistic_regression(df, target, features)
            else:
                result = {"error": "не указаны target или features"}

        elif method == "kmeans":
            features = params.get("features", [])
            n_clusters = params.get("n_clusters", 3)
            if features:
                result = run_kmeans(df, features, n_clusters)
            else:
                result = {"error": "не указаны features для кластеризации"}

        elif method == "ttest":
            column = params.get("column")
            group_column = params.get("group_column")
            if column and group_column:
                result = run_ttest(df, column, group_column)
            else:
                result = {"error": "не указаны column или group_column"}

        elif method == "descriptive":
            column = params.get("column")
            if column:
                result = run_descriptive(df, column)
            else:
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                if numeric_cols:
                    result = run_descriptive(df, numeric_cols[0])
                else:
                    result = {"error": "нет числовых столбцов"}

        else:
            result = {"error": f"неизвестный метод: {method}"}

    except Exception as e:
        result = {"error": str(e)}

    return {
        **state,
        "analysis_result": {
            "params": params,
            **result
        }
    }
