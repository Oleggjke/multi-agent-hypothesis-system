import seaborn as sns

df = sns.load_dataset("titanic")
df.to_csv("data/dataset.csv", index=False)
print("Датасет сохранён!")
