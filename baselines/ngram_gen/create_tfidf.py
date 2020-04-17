"""
Use `ngram_controlled_gen` to create tfidf across all genres.

Use these for clustering later somehow
"""


from ngram_controlled_gen import *
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_tfidf_score(models, all_genres, word):
    """Get TF-IDF score.
    
    Parameters
    ----------
    models: dict where keys are elements of all_genres
    all_genres: list
    word: str

    Returns
    ---------
    list of tf-idf scores of word across all genres
    """
    N = len(all_genres)
    freq = np.array( [models[genre].vocab.counts[word] for genre in all_genres] )
    # print("Freq", freq)
    tf = np.log10(freq + 1)
    # print("TF", tf)
    idf = np.log10(N / np.count_nonzero(freq))
    # print("IDF", idf)
    # print("TF-IDF", tf*idf)
    return list(tf*idf)


def create_tfidf_embeddings():
    """Create embeddings."""
    sg = ScriptGram(n=2)
    sg.load_models()
    # print("genres", sg.all_genres)
    df = pd.DataFrame(columns=["token"]+sg.all_genres)
    global_vocab = set()    
    file_name = os.path.join(GEN_DIR, "tfidf_all_genres.csv")
    # get all vocabs in this order
    for genre in tqdm(sg.all_genres): 
        gvocab = sg.models[genre].vocab.counts
        count = 0
        for word in tqdm(gvocab):
            if word in global_vocab: continue
            count += 1
            global_vocab.add(word)
            scores = get_tfidf_score(sg.models, sg.all_genres, word)
            # df.append(scores)
            df.loc[len(df)] = [word]+scores
        if os.path.exists(file_name): 
            df.to_csv(file_name, mode='a', header=True)
            print("created:", file_name)
            print("added vocabs from:", genre, "; count", count)
        else:
            df.to_csv(file_name, mode='a', header=False)
            print("added vocabs from:", genre, "; count", count)
    

if __name__ == "__main__":
    create_tfidf_embeddings()
    
