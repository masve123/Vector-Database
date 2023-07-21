import numpy as np
import distance
import lsh

import numpy as np
import transformers # this will download from the hugging face model repository
import pickle
import tqdm


import torch # yes, this is just for type-checking

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
model = transformers.BertModel.from_pretrained("bert-base-uncased")

def bert_embedding(sentence: str) -> torch.Tensor:
    """
    convert a string sentence into a BERT embedding
    """

    inputs = tokenizer(
        sentence,
        return_tensors = "pt",
        padding = "max_length",
        max_length = 100,
        truncation = True
    )
    outputs = model(**inputs)
    return np.array(outputs.last_hidden_state.detach().numpy().flatten())


class VectorDatabase:
        """Vector database using locality sensitive hashing
        This class encompasses the entirety of the vector database
        You are free to use whatever distance measure you want
        including cosine distance, euclidean distance, etc."""

    
        def __init__(self,
        vectors: np.array = np.empty((1, 100, 768), dtype = np.float32),
        embedding: object = bert_embedding,
        distance: object = distance.cosine_distance 
    ) -> None:
               self.embedding = embedding
               self.distance = distance
               self.lsh = lsh.LSH()



        def insert(self, obj: object, key_fn: callable = None) -> None:
            if key_fn is None:
                key_fn = self.embedding
            vector = key_fn(obj).astype(np.float32)
            self.lsh.index(vector, obj)


    
        def query(self, sentence: str, top_k: int = 5) -> list:
            vector = self.embedding(sentence).astype(np.float32)
            candidates = self.lsh.query(vector)
            
            # Compute distances for each candidate
            distances = [(candidate, distance) for candidate, _, distance in candidates]

            # Sort the candidates based on the distance
            sorted_candidates = sorted(distances, key=lambda x: x[1])
            
            # Return the top_k closest candidates (i.e., with the smallest distances)
            return [candidate for candidate, _ in sorted_candidates[:top_k]]



        def from_source(self, source: list, key_fn: object = lambda x: x) -> None:
                for ref in (bar := tqdm.tqdm(source)):
                    vector = self.embedding(key_fn(ref)).astype(np.float32)
                    self.lsh.index(vector, ref)
                    bar.set_description("from_source")

        def save(self, filename: str) -> None:
            with open(filename, "wb") as handler:
                pickle.dump(self.lsh, handler) 

        def load(self, filename: str) -> None:
            with open(filename, "rb") as handler:
                self.lsh = pickle.load(handler)


if __name__ == "__main__":
    db = VectorDatabase()

    sentences = [
        "This is a test sentence.",
        "I am another sentence.",
        "Hello world.",
        "The weather is good today.",
        "I like pizza."
    ]

    for sentence in sentences:
        db.insert(sentence) 

    query_sentence = "I love food."
    print(db.query(query_sentence, top_k=2))  # It should print the two most similar sentences to "I love food."






