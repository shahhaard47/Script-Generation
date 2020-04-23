"""
Evaluate generated text with word2vec and GPT2 embeddings (relative to seed_words)
"""

import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from gensim.utils import simple_preprocess
from visualize_word_embeddings import get_gpt_tokenizer
from scipy.spatial import distance
import os
from time import time
import numpy as np
import operator
import torch
import pandas as pd
import csv
from tqdm import tqdm

this_file = os.path.dirname(os.path.abspath(__file__))
keys = ['<Comedy>', '<Action>', '<Adventure>', '<Crime>', '<Drama>', '<Fantasy>', '<Horror>', '<Music>', '<Romance>', 'Sci-Fi', '<Thriller>']
referenceWords = {
    "Action": 'military,battle,shoot,punch,resistance,war,conflict,execute,prosecution,force'.split(','),
    "Comedy": 'humor,clowning,funny,laughter,joke,sarcasm,goofy,playful,hilarious,silly'.split(','),
    "Adventure": 'risk,hazard,chance,journey,voyage,venture,epic,exploring,superhero,adventurers'.split(','),
    "Crime": 'theft,murder,felony,rape,fraud,victim,police,corrupt,kill,violence'.split(','),
    "Drama": 'narrative,emotional,depth,relationship,issues,feelings,grief,culture,tension,theatrical'.split(','),
    "Fantasy": 'fancy,supernatural,dream,imagine,vision,delusion,fairyland,love,epic,imaginary'.split(','),
    "Horror": 'fancy,supernatural,dream,imagine,vision,delusion,fairyland,love,epic,imaginary'.split(','),
    "Music": 'soundtrack,studio,singer,superstar,star,theater,acoustic,composer,copyrighted,celebrity'.split(','),
    "Romance": 'love,affair,darling,flirt,relationship,sentimental,sex,affection,beauty,affection'.split(','),
    "Sci-Fi" : 'future,dystopian,extraterrestrial,psychology,supernatural,scientific,robot,technology,universe,time'.split(','),
    "Thriller": 'suspense,epic,satire,spoof,gangster,adrenalin,excitement,goosebumps,nerve,rivetting'.split(',')
}


class Eval(object):
    def __init__(self):
        self.refWordsEmbeds = [
            self.embedSequence(referenceWords[s]) for s in referenceWords
        ]

        self.refWordsMeans = np.array([
            self.embedSequence(referenceWords[s]).mean(axis=0) for s in referenceWords
        ])
        self.refWordsSums = np.array([
            self.embedSequence(referenceWords[s]).sum(axis=0) for s in referenceWords
        ])

        self.genre = [g for g in referenceWords]
        # (N, T, D) ==> genre, words in genre, dimensin of embedding

    def embedToken(self, word):
        """Different for Word2Vec and GPT2."""
        pass

    def tokenizeSequence(self, sequence):
        """Different for Word2Vec and GPT2."""
        pass

    def embedSequence(self, sequence):
        pass

    def sumTokenEmbeds(self, wordList):
        """
        Takes in list(str) and outputs sum of words' embeddings.
        """
        lst = [self.embedToken(word) for word in wordList]
        lst = np.array(lst)
        return lst.sum(axis=0)

    def averageTokenEmbeds(self, wordList):
        """
        Takes in list(str) and outputs mean of words' embeddings.
        """
        lst = [self.embedToken(word) for word in wordList]
        lst = np.array(lst)
        return lst.mean(axis=0)

    def cosine(self, v1, v2):
        """Get cosine distance between two vectors of same dim."""
        if v1.shape != v2.shape:
            print("cannot calculate distance across different shapes", v1.shape, ",", v2.shape)
            return None
        return distance.cosine(v1, v2)

    def getDist(self, item1, item2):
        """Get distance between embed/token to embed/token.
        Uses self.cosine
        """
        if isinstance(item1, str):
            item1 = self.embedToken(item1)
        if isinstance(item2, str):
            item2 = self.embedToken(item2)
        return self.cosine(item1, item2)

    def embedSequence(self, sequence):
        """
        if sequence is str : then tokenize using gensim.util.simple_preprocess
        otherwise : assume sequence is list(str) and embed each str in list
        Returns 2D numpy array of (T, D) where T is number of tokens and D is dimension
        """
        # tokenize
        if isinstance(sequence, str):
            tokens = self.tokenizeSequence(sequence)
        else:
            # assume its a list if not a str
            tokens = sequence
        # call embed on in loop
        embeds = np.array([self.embedToken(t) for t in tokens])
        return embeds

    def distanceToReferenceWords(self, sequence, genre=None, useSum=True):
        """Gets distance between mean/sum of each category and mean/sum of sequence.
        if genre is None: output list of (genre, prob) with genre of lowest distance first
        if genre is str: return (genre, prob) tuple

        useSum = True because sum makes more sense for combined meaning of words in sequence
        """
        if isinstance(sequence, str):
            # tokenize sequence
            sequence = self.tokenizeSequence(sequence)
        dists = []
        if not useSum:
            # compare means
            inp_seq = self.averageTokenEmbeds(sequence)
            refWords = self.refWordsMeans
            # print(inp_seq.shape)
            # print(self.refWordsMeans.shape)
        else:
            # compare sums
            inp_seq = self.sumTokenEmbeds(sequence)
            # refWords = self.refWordsSums
            refWords = self.refWordsMeans # Just to test (MIXED) sum inp sentence but average seeds
            # print(inp_seq.shape)
            # print(self.refWordsSums.shape)
        if genre is not None:
            try:
                idx = self.genre.index(genre)
            except ValueError:
                print("genre", genre, "key not among reference words")
                return None
            return (genre, self.getDist(inp_seq, refWords[idx]))
        # if no genre is passed
        for i, genre in enumerate(self.genre):
            dists.append((genre, self.getDist(inp_seq, refWords[i])))
            # print(dists[-1])
        dists.sort(key=operator.itemgetter(1))
        # for p in dists:
        #     print(p)
        return dists

class GPT2EmbeddingsEval(Eval):
    def __init__(self):
        """GPT2 Embeddings Eval."""
        self.model, self.tokenizer = get_gpt_tokenizer()
        self.embeddings = self.model.transformer.wte.weight
        Eval.__init__(self)

    def embedToken(self, word):
        """Different for Word2Vec and GPT2."""
        tmp = torch.mean(self.embeddings[self.tokenizer.encode(word), :], 0).unsqueeze(0).data.cpu().numpy()[0]
        return tmp

    def tokenizeSequence(self, sequence):
        """Different for Word2Vec and GPT2."""
        tokens = self.tokenizer.tokenize(sequence)
        return tokens


class Word2VecEval(Eval):
    def __init__(self):
        """Word2Vec model."""
        corpus = "word2vec-google-news-300"
        print("Initializing", corpus, "...")
        tic = time()
        self.model = api.load(corpus)
        self.wordVector = self.model.wv
        print("loaded model; time :", time() - tic) 
        Eval.__init__(self)

    def embedToken(self, word):
        """Returns ndarray of shape (D,)"""
        emb = None
        try:
            emb = self.model[word]
        except:
            pass
        return emb


    def _filter_tokens_not_in_model(self, tokens):
        """check if each token appears in w2v vocab."""
        new_toks = []
        for t in tokens:
            if t in self.wordVector:
                new_toks.append(t)
        return new_toks

    def tokenizeSequence(self, sequence):
        """Take in sentence, outputs list of str tokens."""
        toks = simple_preprocess(sequence)
        return self._filter_tokens_not_in_model(toks)

    

def get_dist_closest(model, text, genre):
    dists = model.distanceToReferenceWords(text, useSum=True) 
    closest = "yes" if genre == dists[0][0] else dists[0][0]
    for d in dists:
        if genre == d[0]:
            return str(d[1]), closest
    # should never get here
    print("Something went wrong, couldn't find genre")
    exit()

gpt = GPT2EmbeddingsEval()
wv = Word2VecEval()

# Evaluate data/output.csv
df = pd.read_csv('data/output.csv')

# 'data/seed_text.csv'
# seed_df = pd.DataFrame(referenceWords)
# seed_df.to_csv('data/seed_text.csv', index=False)

# *_closest_label = "yes" if same as genre, else correct label

fname = "data/output_evals_2.csv"
with open(fname, "w") as f:
    csv_out = csv.writer(f)
    csv_out.writerow(('genre', 'seed_text', 'w2v_lexical_style_GPT', 'w2v_closest_label_GPT', 'w2v_lexical_style_bigram', 'w2v_closest_label_bi', 'w2v_lexical_style_trigram', 'w2v_closest_label_tri', 'gpt_lexical_style_GPT', 'gpt_closest_label_GPT', 'gpt_lexical_style_bigram', 'gpt_closest_label_bi', 'gpt_lexical_style_trigram', 'gpt_closest_label_tri'))

    for i in tqdm(range(len(df))):
        genre = df.iloc[i, 0].replace("<", "").replace(">", "") #
        seed_text = df.iloc[i, 1] #
        gpt_gen = df.iloc[i, 2]
        bi = df.iloc[i, 3]
        tri = df.iloc[i, 4]
        w2v_gpt, w2v_c_gpt = get_dist_closest(wv, gpt_gen, genre) #
        w2v_bi, w2v_c_bi = get_dist_closest(wv, bi, genre) #
        w2v_tri, w2v_c_tri = get_dist_closest(wv, tri, genre) #
        gpt_gpt, gpt_c_gpt = get_dist_closest(gpt, gpt_gen, genre) #
        gpt_bi, gpt_c_bi = get_dist_closest(gpt, bi, genre) #
        gpt_tri, gpt_c_tri = get_dist_closest(gpt, tri, genre) #
        csv_out.writerow((genre, seed_text, w2v_gpt, w2v_c_gpt, w2v_bi, w2v_c_bi, w2v_tri, w2v_c_tri, gpt_gpt, gpt_c_gpt, gpt_bi, gpt_c_bi, gpt_tri, gpt_c_tri))


# model = Word2Vec(model)


comedy = ("Comedy", """ARTHUR
Well, that's just a little thin.
LIONEL
Yeah, just a little thin maybe.
Arthur sits down next to the desk.
ARTHUR
It's a little bit of a joke, but,
still there's a lot of funny jokes to the people.""")

action = ("Action", """wall.
MAN
What the hell's that?
INT. PLATFORM - NIGHT
In a loud mixture of seats, from the express train. The still sounds as if in the whole plane. Nevertheless, the growing chorus of the passengers impatiently.
MAN (o.s.)
Hey, I've got a column on that you know. It's just a first
check of the fuel line, Ray. That's it.""")

horror = ("Horror", """wall.
CAMERA PANS TO A CLOSE SHOT of the man. He starts to walk slowly.
INSERT - CLOSE ON SIDNEY'S EXPRESSION AREA, which is dropped off of her fingers.
A HAND plunges INTO FRAME between the clasped side and in the man's hand. He is holding his head. He looks up slowly, slowly.""")

fantasy = ("Fantasy", """and the KNOCKS on the
ground, then turns and runs toward the open crypt. INT. COLD WILLOW'S STUDY - NIGHT
In the background, the door to the small rooms of the
Old Brewery occupies the ancient temple. He walks through the doorway and out into the corridor.""")


