class BioBert2Vect(object):

    error_message = """
    This is unnecessary. This can be replaced by:
    from sentence_transformers import SentenceTransformer
    biobert = SentenceTransformer("dmis-lab/biobert-base-cased-v1.1")
    embeddings =  biobert.encode(sentences)
    """

    raise DeprecationWarning(error_message)

    def __init__(self) -> None:
        "Load BioBERT"
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.1"
        )
        self.model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    def get_sentence_embedding(self, sentence, method: str = "last_hidden_state"):
        """Takes a list of sentences and converts them to BioBERT embeddings"""
        import torch

        # Tokenize input text
        inputs = self.tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True, max_length=128
        )

        # Get model output
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract the embeddings from the model output
        valid_methods = ["last_hidden_state", "pooler_output"]
        if method not in valid_methods:
            raise NotImplementedError(
                f"{method} method not implemented. Select from {valid_methods}"
            )

        if method == "last_hidden_state":
            embeddings = torch.mean(outputs.last_hidden_state, dim=1)

        elif method == "pooler_output":
            embeddings = outputs.pooler_output

        embeddings = embeddings.numpy()
        return embeddings

    def get_word_embeddings(self, sentence):
        # Tokenize input text
        inputs = self.tokenizer(
            **sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the embeddings from the model output
        embeddings = outputs.last_hidden_state

        # Convert to numpy array
        embeddings = embeddings.numpy()

        return embeddings


if __name__ == "__main__":

    sentences = [
        "Cancer is a group of diseases involving abnormal cell growth with the potential to invade or spread to other parts of the body.",
        "Protein folding is the physical process by which a protein structure assumes its functional three-dimensional structure from a linear chain of amino acids.",
    ]

    from sentence_transformers import SentenceTransformer

    biobert = SentenceTransformer("dmis-lab/biobert-base-cased-v1.1")
    embeddings = biobert.encode(sentences)

    # biobert2vect = BioBert2Vect()
    # embeddings = biobert2vect.get_sentence_embedding(
    #     sentences, method="last_hidden_state"
    # )

    # for sentence, embedding in zip(sentences, embeddings):
    #     print(f"Sentence: {sentence}")
    #     print(f"Embedding shape: {embedding.shape}")
    #     print(f"Embedding: {embedding}")
    #     print()
