import argparse
import pandas as pd

def precision(references, predictions):
    total = len(predictions)
    numerator = set(predictions).intersection(set(references))
    return len(numerator) / total

def recall(references, predictions):
    total = len(references)
    numerator = set(references).intersection(set(predictions))
    return len(numerator) / total

def f1(precision, recall):
    return 2 * precision * recall / (precision + recall)


def main(args):
    
    # read in the data
    df = pd.read_csv(args.input_tsv, sep="\t")
    
    for i, row in df.iterrows():
        
        if (row[args.y] is None) or (row[args.yhat]):
            continue
        
        references = row[args.y].astype(str).tolist()
        predictions = row[args.yhat].astype(str).tolist()
        df.loc[i, "precision"] = precision(references, predictions)
        df.loc[i, "recall"] = recall(references, predictions)
        df.loc[i, "f1"] = f1(df.loc[i, "precision"], df.loc[i, "recall"])
    
    # write the results to a tsv file
    print(f"Writing results to {args.output_tsv}")
    df.to_csv(args.output_tsv, sep="\t")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="trialgpt questioner evaluation")
     
    parser.add_argument(
        "-i",
        "--input_tsv",
        type=str,
        help="Path to the input tsv file",
    )
    
    parser.add_argument(
        "-o",
        "--output_tsv",
        type=str,
        help="Path to the output tsv file",
    )
    
    parser.add_argument(
        "-y",
        type=str,
        help="gold standard answer field name",
    )
    
    parser.add_argument(
        "-yhat",
        type=str,
        help="answer to be evaluated field name",
    )
    
    args = parser.parse_args()
    main(args)