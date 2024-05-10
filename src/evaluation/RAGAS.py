import argparse
import os
import sys

import pandas as pd
from langchain_text_splitters import RecursiveJsonSplitter
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator

####### Add src folder to the system path so it can call utils
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current script
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from utils.utils import connect_to_mongoDB


# TODO: Remove nest_asyncio if not necessary
import nest_asyncio
nest_asyncio.apply()


def str_format(x:str)->str:
    x =  (x.replace('"', '')
          .replace("\\n"," ")
          .replace("\n"," ")
          .replace("  ", " ")
          .replace("{","")
          .replace("}", "")
          .replace("[","")
          .replace("],",";")
          .replace("]","")
          )
    return x

def create_documents(args):

    studies = []
    # Connect to mongoDB and get n studies
    with connect_to_mongoDB(args.user, args.pwd) as client:
        db = client[args.db]
        collection = db[args.collection]
        
        #Remove Locations from the document
        collection.update_many({}, {"$unset": {"protocolSection.contactsLocationsModule": 1}})

        print(
            f"Pulling first {args.n} CT studies from MongoDB db:{args.db} collection:{args.collection}"
        )
        for study in collection.find({}).limit(args.n):
            nctId = study.pop("_id")
            studies.append({nctId: study})

    # Use langchain's RecursiveJsonSplitter to generate document chunks
    print(f"Generating chunks max size {args.size}")
    splitter = RecursiveJsonSplitter(max_chunk_size=args.size)
    docs = splitter.create_documents(texts=studies)
    # Add the nctid as file name for all chunks.
    # this it needed for the RAGAS lib.
    for d in docs:
        d.metadata["filename"] = d.page_content[2:13]
        d.page_content = str_format(d.page_content)
        
    return docs


def set_llms(args):
    # Set the LLM
    if args.hf:  # if HuggingFace token provided
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = args.hf
        from langchain_community.llms import VLLMOpenAI
        
        generator_llm = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=f"{args.host}:{args.ports[0]}/v1/",
            model_name=args.generator,
            model_kwargs={"stop": ["."]},
            )
        
        critic_llm = VLLMOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=f"{args.host}:{args.ports[1]}/v1/",
            model_name=args.critic,
            model_kwargs={"stop": ["."]},
            )
        
    else:  # Else assumes that use Ollama
        from langchain_community.llms import Ollama

        generator_llm = Ollama(model=args.generator)
        critic_llm = Ollama(model=args.critic)

    return generator_llm, critic_llm


def main(args, verbose:bool=False):
    
    os.environ["LANGCHAIN_PROJECT"] = "RAGAS"
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if args.ls else "false"

    docs = create_documents(args)
    
    generator_llm, critic_llm = set_llms(args)
    
    if args.embeddings == "GPT4All":
        from langchain_community.embeddings import GPT4AllEmbeddings
        embeddings = GPT4AllEmbeddings()
    
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=args.embeddings)
        
    generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)
    
    if verbose:
        print("Generating RAG evaluation data using RAGAS:")
        print(f"- Generator: {args.generator.split('/')[-1]}")
        print(f"- Critic: {args.critic.split('/')[-1]}")
        print(f"- Embeddings: {args.embeddings.split('/')[-1]}")
        print(
            f"- Eval dataset: size {args.test_size} ",
            f"({args.simple*100:.2f}% simple, ",
            f"{args.reasoning*100:.2f}% reasoning, ",
            f"{args.multi_context*100:.2f}% multi context)",
        )
        print(f"Number of documents {len(docs)}")
    
    eval_ds = generator.generate_with_langchain_docs(
        docs,
        test_size=int(args.test_size),
        distributions={
            simple: args.simple,
            reasoning: args.reasoning,
            multi_context: args.multi_context,
        },
        raise_exceptions=True,
        is_async=True # as per https://github.com/explodinggradients/ragas/issues/709
    )
    eval_ds.to_pandas().to_csv(args.output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate RAG evaluation dataset")
    parser.add_argument("output", type=str, help="path to output file")
    # Mongo DB connection settings
    parser.add_argument("-user", type=str, help="MongoDB user name")
    parser.add_argument("-pwd", type=str, help="MongoDB password")
    parser.add_argument("-app", type=str, default="cluster0", help="MongoDB cluster")
    parser.add_argument("-db", type=str, default="ctGov", help="MongoDB database name")
    parser.add_argument("-c", "--collection", type=str, help="MongoDB collection name")

    # RAGAS settings
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="Number of CT studies (a.k.a documents)",
    )

    parser.add_argument("-size", type=int, default=2_000, help="Document chunk size")

    parser.add_argument(
        "-ls",
        default=argparse.SUPPRESS,
        help="LangSmith API key for tracking.",
    )

    parser.add_argument(
        "-hf",
        default=argparse.SUPPRESS,
        help="HuggingFace Token. If not provided, assumes that Ollama.",
    )
    
    parser.add_argument(
        "-host",
        type=str,
        default="http://0.0.0.0",
        help="LLM server host.",
    )
    
    parser.add_argument(
        "-ports", type=str, nargs="+", default=["8000", "8001"], help="2 LLM server ports. One for the generator and critic."
    )
    
    
    parser.add_argument(
        "-g",
        "--generator",
        type=str,
        default="mistral",
        help="LLM user as generator E.g. For Ollama use 'mistral', for HF use 'mistralai/Mistral-7B-Instruct-v0.2'",
    )
    parser.add_argument(
        "-cr",
        "--critic",
        type=str,
        default="mistral",
        help="LLM user as critic E.g. For Ollama use 'mistral', for HF use 'mistralai/Mistral-7B-Instruct-v0.2'",
    )

    parser.add_argument(
        "-e",
        "--embeddings",
        type=str,
        default="all-MiniLM-L6-v2",
        help="HuggingFace embeddings'",
    )

    parser.add_argument(
        "-test_size",
        type=int,
        default=10,
        help="Number of question-context-answer triplets",
    )

    parser.add_argument(
        "-s", "--simple", type=float, default=0.5, help="Proportion of simple questions"
    )
    parser.add_argument(
        "-r",
        "--reasoning",
        type=float,
        default=0.25,
        help="Proportion of reasoning questions",
    )
    parser.add_argument(
        "-mc",
        "--multi_context",
        type=float,
        default=0.25,
        help="Proportion of multi context questions",
    )

    parser.set_defaults(hf=False, ls=False)
    args = parser.parse_args()
    main(args, True)
