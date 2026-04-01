import pandas as pd
from scipy import stats

def run_correlation(df: pd.DataFrame, col1: str, col2: str) -> dict:
    # убираем строки с пропусками
    clean = df[[col1, col2]].dropna()

    # коэффициент корреляции Пирсона
    pearson_r, pearson_p = stats.pearsonr(clean[col1], clean[col2])

    # коэффициент корреляции Спирмена
    spearman_r, spearman_p = stats.spearmanr(clean[col1], clean[col2])

    return {
        "method": "correlation",
        "col1": col1,
        "col2": col2,
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 4),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 4),
        "n": len(clean)
    }
