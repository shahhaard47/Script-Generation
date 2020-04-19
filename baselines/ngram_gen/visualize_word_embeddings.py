"""
Use `ngram_controlled_gen` to create tfidf across all genres.

Use these for clustering later somehow
"""


from ngram_controlled_gen import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import math
import re
import nltk
# from gensim.models import word2vec
import matplotlib.pyplot as plt
import pickle
from time import time

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

def calculate_whole_genre(models, all_genres, words):
    df = pd.DataFrame(columns=["token"]+all_genres)
    for word in tqdm(words):
        scores = get_tfidf_score(sg.models, sg.all_genres, word)
        # df.append(scores)
        df.loc[len(df)] = [word]+scores
    return df

def tfidf_async(sg):
    """DOESN'T WORK!!!!"""
    all_genres = sg.all_genres
    df = pd.DataFrame(columns=["token"]+all_genres)
    global_vocab = set()
    for genre in all_genres:
        # do this async
        gvoc = sg.models[genre].vocab.counts
        for word in gvoc:
            global_vocab.add(word)
    # print(len(global_vocab))
    global_vocab = list(global_vocab)
    split = math.ceil(len(global_vocab) / 16)

    tmps = []
    pool = mp.Pool()
    for i in range(16):
        pool.apply_async(calculate_whole_genre, (sg.models, all_genres, global_vocab[i*split:i*split+split]))
    for t in tmps:
        smalldf = t.get()
        df = pd.concat([df, smalldf], axis=0)
    pool.close()
    pool.join()
    df = df.drop_duplicates(subset="token", keep="first")
    return df

def create_tfidf_embeddings():
    """Create embeddings."""
    sg = ScriptGram(n=2)
    sg.load_models()
    # print("genres", sg.all_genres)
    df = pd.DataFrame(columns=["token"]+sg.all_genres)
    global_vocab = set()    
    file_name = os.path.join(GEN_DIR, "tfidf_all_genres0.csv")

    # if False:
    #     par_df = tfidf_async(sg)
    #     par_df.to_csv(file_name, index=False)
    #     exit()
    # get all vocabs in this order
    for genre in tqdm(sg.all_genres): 
        gvocab = sg.models[genre].vocab.counts
        count = 0
        df = df[0:0]
        pbar = tqdm(gvocab)
        pbar.set_description("Processing - " + genre)
        for word in pbar:
            if word in global_vocab: continue
            count += 1
            global_vocab.add(word)
            scores = get_tfidf_score(sg.models, sg.all_genres, word)
            # df.append(scores)
            df.loc[len(df)] = [word]+scores
        if not os.path.exists(file_name): 
            df.to_csv(file_name, mode='a', header=True, index=False)
            print("created:", file_name)
            print("added vocabs from:", genre, "; count", count)
        else:
            df.to_csv(file_name, mode='a', header=False, index=False)
            print("added vocabs from:", genre, "; count", count)
    

def extractFromDF(df, cols=None, thresh=0):
    print("Cleaning up data")
    tic = time()

    if cols is not None:
        token = df.columns.values[0]
        df = df[[token] + list(cols)]
    # remove zero sum rows
    mask = df.iloc[:, 1:].sum(axis=1) > 0
    df = df[mask]
    # print(df)
    # normalize df
    sum_df = df.iloc[:, 1:].sum(axis=1)
    normalized_df = df.copy()
    normalized_df.at[:, 1:] = df.iloc[:, 1:].div(sum_df, axis=0)
    # print(df)
    # print(normalized_df)
    # threshold
    tmask = normalized_df.iloc[:, 1:].sum(axis=1) > thresh
    normalized_df = normalized_df[tmask]
    print("Done", time() - tic)
    # print(normalized_df)

    # print(new_df.iloc[:, 0])
    labels = normalized_df.iloc[:, 0].tolist()
    print("labels", len(labels))
    tokens = normalized_df.iloc[:, 1:].values.tolist()
    # print("tokens", tokens.shape)
    return (labels, tokens)

def getColors(df, cols=None, thresh=0):
    labels, tokens = extractFromDF(df, cols, thresh)
    tokens = np.array(tokens)
    colors = np.argmax(tokens, axis=1)
    cdict = {}
    if cols is None: cols = df.columns.values[1:]
    for i, genre in enumerate(cols):
        cdict[i] = genre
    return colors.tolist(), cdict

def fit_tsne_model(df, parallel=True, cols=None,
                    thresh=0, 
                    pickleFileName="saved_tsne_values.pickle",
                    cont_saved=False):
    """Returns embeddings from data and labels in df.
    if cont_saved is True:
        cols, thresh - arguments are ignored
    `parallel` : arg is a misnomer (True: is used with method_2 and False with method_1)
    """
    labels, tokens = extractFromDF(df, cols=cols, thresh=thresh)

    if parallel: # Method 2
        from MulticoreTSNE import MulticoreTSNE
        tokens = np.array(tokens)
        labels = np.array(labels)
        tsne_model = MulticoreTSNE(perplexity=40, n_components=2,
                      init='random', n_iter=2500, random_state=23, n_jobs=16, n_iter_without_progress=50)
    else: # Method 1
        from sklearn.manifold import TSNE
        tsne_model = TSNE(perplexity=40, n_components=2,
                      init='pca', n_iter=2500, random_state=23, n_jobs=16, n_iter_without_progress=50)

    if cont_saved:
        print('loading saved tsne values from', pickleFileName)
        tic = time()
        with open(pickleFileName, 'rb') as f:
            new_values = pickle.load(f)
        print("Done", time() - tic)
    else:
        print('fitting tsne model')
        tic = time()
        new_values = tsne_model.fit_transform(tokens)
        print("fitted values", new_values.shape)
        print("Done", time() - tic)
        print('saving tsne values to', pickleFileName)
        # return new_values
        tic = time()
        with open(pickleFileName, 'wb') as f:
            pickle.dump(new_values, f)
        print("Done", time() - tic)
          
    return new_values, labels
    
def create_1_tSNE(df, cols=None, thresh=0, pickleFileName="saved_tsne_values.pickle", cont_saved=True):
    """Create tSNE plot."""

    print("Creating tSNE plot (method 1)")
    new_values, labels = fit_tsne_model(df, parallel=False, cols=cols, thresh=thresh, pickleFileName=pickleFileName, cont_saved=cont_saved)
    
    colors, cdict = getColors(df, cols=cols, thresh=thresh)
    colorNames = [cdict[c] for c in range(len(cdict))]

    x = []
    y = []
    print("Plotting")
    print()
    tic = time()
    for value in tqdm(new_values):
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))

    # scatter = plt.scatter(x, y, c=colors, cmap=plt.cm.get_cmap("jet", len(colorNames)), marker='.')
    plt.scatter(x, y, c=colors, cmap=plt.cm.get_cmap("jet", len(colorNames)), marker='.')
    # for i in tqdm(range(len(x))):
        #     # plt.scatter(x[i], y[i])
        #     plt.scatter(x[i], y[i], c=colors[i], cmap=plt.cm.get_cmap("jet", len(colorNames)), marker='.')
        #     plt.annotate(labels[i],
        #                  xy=(x[i], y[i]),
        #                  xytext=(5, 2),
        #                  textcoords='offset points',
        #                  ha='right',
        #                  va='bottom')
    bar = plt.colorbar(ticks=range(len(colorNames)))
    bar.set_ticklabels(colorNames)
    
    print("Done", time() - tic)
    plt.show()


def create_2_tSNE(df, cols=None, thresh=0, pickleFileName="saved_tsne_values.pickle", cont_saved=True):
    """Worse than create_1."""
    print("Creating tSNE plot (method 2)")
    new_values, labels = fit_tsne_model(
        df, parallel=True, cols=cols, thresh=thresh, pickleFileName=pickleFileName, cont_saved=cont_saved)
    vis_x = new_values[:, 0]
    vis_y = new_values[:, 1]

    print("plotting setup")
    tic = time()
    plt.scatter(vis_x, vis_y, cmap=plt.cm.get_cmap("jet", 10), marker='.')
    plt.colorbar(ticks=range(10))
    print("Done", time() - time())
    # plt.clim(-0.5, 9.5)
    plt.show()

def get_GPT2_embeddings():
    # 1. Get vocab
    file_name = os.path.join(GEN_DIR, "tfidf_all_genres1.csv")
    df = pd.read_csv(file_name, index_col=False)

    # 2. Use GPT tokenizer to encode 

    # 3. Get embeddings using the ints

    # 4. plot them in similar ways (to tfidf)
            # maybe use tf-idf colorings on embeddings

    pass

if __name__ == "__main__":
    # create_tfidf_embeddings()
    print("reading csv")
    file_name = os.path.join(GEN_DIR, "tfidf_all_genres1.csv")
    df = pd.read_csv(file_name, index_col=False)

    # create_1_tSNE(df, cols=['Action', 'Drama', 'Comedy'], thresh=0.5,pickleFileName="tsne_actionDramaComedy_t0.5.pickle", cont_saved=True)
    # pickleFileName = os.path.join(file_loc, "tsne_all_t0.pickle")
    # create_1_tSNE(df, cols=None, thresh=0.5, pickleFileName=pickleFileName, cont_saved=True)


    # cols = ['Adventure', 'Romance', 'Horror', 'Fantasy']
    cols = ['Biography', 'Film-Noir', 'History', 'Music', 'Short', 'Sport']
    nfile = 'tsne_' + ''.join(cols) + '.pickle'
    pickleFileName = os.path.join(file_loc, nfile) # all > 300 movies
    create_1_tSNE(df, cols=cols, thresh=0, pickleFileName=pickleFileName, cont_saved=True)

"""
Action	    308
Adventure	186
Animation	43
Biography	3
Comedy	    378
Crime	    216
Drama	    626
Family	    44
Fantasy	    119
Film-Noir	4
History	    3
Horror	    158
Music	    5
Musical	    25
Mystery	    111
Romance	    200
Sci-Fi	    169
Short	    3
Sport	    2
Thriller	385
War	        28
Western	    14
"""
