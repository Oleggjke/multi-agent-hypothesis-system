import pandas as pd
from scipy import stats
import numpy as np

def run_descriptive(df: pd.DataFrame, column: str) -> dict:
    clean = df[column].dropna()

    return {
        "method": "descriptive",
        "column": column,
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "std": round(float(clean.std()), 4),
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "n": len(clean)
    }

def run_ttest(df: pd.DataFrame, column: str, group_column: str) -> dict:
    # проверяем что group_column существует как столбец
    if group_column not in df.columns:
        return {"error": f"столбец '{group_column}' не найден в датасете"}

    groups = df[group_column].dropna().unique()

    if len(groups) < 2:
        return {"error": "нужно минимум 2 группы для t-теста"}

    group1 = df[df[group_column] == groups[0]][column].dropna()
    group2 = df[df[group_column] == groups[1]][column].dropna()

    if len(group1) < 2 or len(group2) < 2:
        return {"error": "недостаточно данных в группах"}

    t_stat, p_value = stats.ttest_ind(group1, group2)

    return {
        "method": "ttest",
        "column": column,
        "group_column": group_column,
        "group1": str(groups[0]),
        "group2": str(groups[1]),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "mean_group1": round(float(group1.mean()), 4),
        "mean_group2": round(float(group2.mean()), 4),
        "significant": bool(p_value < 0.05),
        "n1": len(group1),
        "n2": len(group2)
    }
