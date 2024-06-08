import argparse
import pandas as pd
from rouge import Rouge
import bert_score.score as bertscore
import bleurt.score as bleurtscore


def main(args):
                
        print(f"Loading data from {args.input_tsv} for evaluation")
        
        # read in the data
        df = pd.read_csv(args.input_tsv, sep="\t")
        references = df[args.y].tolist()
        predictions = df[args.yhat].tolist()
        
        # Rouge score
        print("Calculating Rouge scores")
        rogue = Rouge()
        rouge_score = rogue(references, predictions)
        
        df["rouge1f_score"] = [score["rouge-1"]["f"] for score in rouge_score]
        df["rouge2f_score"] = [score["rouge-2"]["f"] for score in rouge_score]
        df["rougelf_score"] = [score["rouge-l"]["f"] for score in rouge_score]
        
        # Bert score
        print("Calculating Bert scores")
        bertScore_Precision, bertScore_Recall, bertScore_F1 = bertscore(predictions, references, model_type='microsoft/deberta-xlarge-mnli', lang='en', device ='cpu' , verbose=True, rescale_with_baseline=True) # roberta-large
        bertscores = bertScore_F1.numpy()
            
        ## clip scores to [0,1]
        bertscores = np.array([np.clip(num) for num in bertscores])
        df["bert_score"] = bertscores
        
        # Bleurt score
        print("Calculating Bleurt scores")
        bleurtscorer = bleurtscore.BleurtScorer(checkpoint="BLEURT-20")
        bleurtscores = bleurtscorer.score(references=references, candidates=predictions, batch_size =1)
        df["bleurt_score"] = bleurtscores
        
        # write the results to a tsv file
        print(f"Writing results to {args.output_tsv}")
        df.to_csv(args.output_tsv, sep="\t")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="ct.gov questioner evaluation")
     
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