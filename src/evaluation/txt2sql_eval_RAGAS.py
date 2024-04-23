import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness, context_entity_recall, faithfulness

SQL_syntaxt = [
    "SELECT ",
    " IN ",
    " FROM ",
    " WHERE ",
    " WHEN ",
    " COUNT ",
    "distinct ",
    "DISTINCT ",
    "GROUP BY",
    " JOIN ",
    " ON ",
    " AS ",
    " COUNT ",
    " CAST ",
    "SUBSTRING",
    "COALESCE",
    "ARRAY_REMOVE",
    " ARRAY ",
    " CASE ",
    " WHEN ",
    " THEN ",
    " END ",
    " IF ",
    "NULL",
    "SUM",
    "[",
    "]",
    "(",
    ")",
    " LIKE ",
    " ILIKE ",
    ".",
    ",",
    "  ",
]

results_dir = "./results/txt2sql/"
inp_file_name = "dspy.Mistral-7B-Instruct-v0.2.eval.tsv"
out_file_name = "dspy.Mistral-7B-Instruct-v0.2.eval.RAGAS.tsv"

df = pd.read_csv(f"{results_dir}{inp_file_name}", sep="\t", index_col=0)

tmp = df[["question", "gold_std_query", "llm_query"]]
tmp = tmp.rename(columns={"gold_std_query": "ground_truth", "llm_query": "answer"})
tmp.fillna("NaN", inplace=True)
tmp = {k: list(tmp[k].values) for k in tmp.columns}

dataset = Dataset.from_dict(tmp)
score = evaluate(dataset, metrics=[answer_correctness])
df["sql_query_correctness"] = score.to_pandas()["answer_correctness"]


tmp = df[["gold_std_query", "llm_query"]]
for c in tmp.columns:
    for s in SQL_syntaxt:
        tmp[c] = tmp[c].str.replace(s, " ")

tmp = tmp.rename(columns={"gold_std_query": "contexts", "llm_query": "ground_truth"})
tmp.fillna("NaN", inplace=True)
tmp["contexts"] = tmp["contexts"].apply(lambda x: x.split("Random_String"))
tmp = {k: list(tmp[k].values) for k in tmp.columns}

dataset = Dataset.from_dict(tmp)
score = evaluate(dataset, metrics=[context_entity_recall])
df["sql_query_recall"] = score.to_pandas()["context_entity_recall"]


tmp = df[["question", "gold_std_output", "llm_output"]]
tmp.fillna("NaN", inplace=True)
tmp = tmp.rename(columns={"gold_std_output": "answer", "llm_output": "ground_truth"})
tmp = {k: list(tmp[k].values) for k in tmp.columns}

dataset = Dataset.from_dict(tmp)
score = evaluate(dataset, metrics=[answer_correctness])
df["sql_output_correctness"] = score.to_pandas()["answer_correctness"]

tmp = df[["question", "llm_output", "llm_answer"]]
tmp = tmp.rename(columns={"llm_output": "contexts", "llm_answer": "answer"})
tmp["contexts"] = tmp["contexts"].apply(lambda x: x.split("Random_String"))
tmp = {k: list(tmp[k].values) for k in tmp.columns}

dataset = Dataset.from_dict(tmp)
score = evaluate(dataset, metrics=[faithfulness])
df["answer_faithfulness"] = score.to_pandas()["faithfulness"]

df.to_csv(results_dir + out_file_name, sep="\t", index=False)
