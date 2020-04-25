"""
Evaluate generated text with word2vec and GPT2 embeddings (relative to seed_words)
"""

from operator import itemgetter
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from gensim.utils import simple_preprocess
from visualize_words import get_gpt_tokenizer
from scipy.spatial import distance
import os
from time import time
import numpy as np
import operator
import torch
import pandas as pd
import csv
from tqdm import tqdm
import nltk
try:
    from nltk.corpus import stopwords
except:
    print("downloading stopwords")
    nltk.download('stopwords')
    from nltk.corpus import stopwords

# Setting up paths
file_loc = os.path.dirname(os.path.abspath(__file__))
HOME_dir = os.path.dirname(os.path.dirname(file_loc))

DATA_DIR = os.path.join(HOME_dir, 'data')

this_file = os.path.dirname(os.path.abspath(__file__))
keys = ['<Action>', '<Comedy>', '<Thriller>', '<Horror>', '<Romance>', '<Sci-Fi>', '<Fantasy>']
# 'Action', 'Comedy', 'Thriller', Horror, Romance, Romance, Sci-Fi, Fantasy
referenceWords = {
    "Action": 'military,battle,shoot,punch,resistance,war,conflict,execute,prosecution,force'.split(','),
    "Comedy": 'humor,clowning,funny,laughter,joke,sarcasm,goofy,playful,hilarious,silly'.split(','),
    # "Adventure": 'risk,hazard,chance,journey,voyage,venture,epic,exploring,superhero,adventurers'.split(','),
    # "Crime": 'theft,murder,felony,rape,fraud,victim,police,corrupt,kill,violence'.split(','),
    # "Drama": 'narrative,emotional,depth,relationship,issues,feelings,grief,culture,tension,theatrical'.split(','),
    "Fantasy": 'fancy,supernatural,dream,imagine,vision,delusion,fairyland,love,epic,imaginary'.split(','),
    "Horror": 'fancy,supernatural,dream,imagine,vision,delusion,fairyland,love,epic,imaginary'.split(','),
    # "Music": 'soundtrack,studio,singer,superstar,star,theater,acoustic,composer,copyrighted,celebrity'.split(','),
    "Romance": 'love,affair,darling,flirt,relationship,sentimental,sex,affection,beauty,affection'.split(','),
    "Sci-Fi" : 'future,dystopian,extraterrestrial,psychology,supernatural,scientific,robot,technology,universe,time'.split(','),
    "Thriller": 'suspense,epic,satire,spoof,gangster,adrenalin,excitement,goosebumps,nerve,rivetting'.split(',')
}


class Eval(object):
    def __init__(self):
        self.genre = [g for g in referenceWords]

        self.refWordsEmbeds = np.array([
            self.embedSequence(referenceWords[g]) for g in self.genre
        ])

        # self.refWordsMeans = np.array([
        #     self.embedSequence(referenceWords[s]).mean(axis=0) for s in referenceWords
        # ])
        # self.refWordsSums = np.array([
        #     self.embedSequence(referenceWords[s]).sum(axis=0) for s in referenceWords
        # ])

        self.stop_words = set(stopwords.words('english'))

        self.k = 10
        # print(self.stop_words)
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

    def removeStopWords(self, sequence):
        """Split sequence by space, and removes stop words.
        sequence must be str
        Returns str with stop words removed"""
        if not isinstance(sequence, str):
            print("not a str, can't remove stop words")
            return sequence
        tmp = sequence.split(' ')
        new_seq = []
        for t in tmp:
            if t not in self.stop_words: new_seq.append(t)
        return ' '.join(new_seq)

    def distanceToReferenceWords(self, sequence):
        """Gets distance between mean/sum of each category and mean/sum of sequence.
        return dict {genre : prob}

        useSum = True because sum makes more sense for combined meaning of words in sequence
        """
        if isinstance(sequence, str):
            # tokenize sequence
            sequence = self.tokenizeSequence(sequence)
        # remove stop_words
        prob = []
        for itr, tok in enumerate(sequence):
            # get distance 
            # print(self.refWordsEmbeds.shape)
            tok = self.embedToken(tok)
            a, b, c = self.refWordsEmbeds.shape
            dists = np.apply_along_axis(self.getDist, axis=2, arr=self.refWordsEmbeds, item2=tok)
            dists = dists.reshape(-1)
            min_idx = np.argpartition(dists, self.k)[:self.k]
            min_idx = np.unravel_index(min_idx, (a, b))
            p = np.zeros(a)
            for idx in min_idx[0]:
                p[idx] += 1
            prob.append(p/a)
        prob = np.array(prob).mean(axis=0)
        dists = []
        prob_dict = {}
        for i, g in enumerate(self.genre):
            prob_dict[g] = prob[i]
        return prob_dict

        # if not useSum:
            #     # compare means
            #     inp_seq = self.averageTokenEmbeds(sequence)
            #     refWords = self.refWordsMeans
            #     # print(inp_seq.shape)
            #     # print(self.refWordsMeans.shape)
            # else:
            #     # compare sums
            #     inp_seq = self.sumTokenEmbeds(sequence)
            #     # refWords = self.refWordsSums
            #     refWords = self.refWordsMeans # Just to test (MIXED) sum inp sentence but average seeds
            #     # print(inp_seq.shape)
            #     # print(self.refWordsSums.shape)
            # if genre is not None:
            #     try:
            #         idx = self.genre.index(genre)
            #     except ValueError:
            #         print("genre", genre, "key not among reference words")
            #         return None
            #     return (genre, self.getDist(inp_seq, refWords[idx]))
            # # if no genre is passed
            # for i, genre in enumerate(self.genre):
            #     dists.append((genre, self.getDist(inp_seq, refWords[i])))
            #     # print(dists[-1])
            # dists.sort(key=operator.itemgetter(1))
            # # for p in dists:
            # #     print(p)
            # return dists

class GPT2EmbeddingsEval(Eval):
    def __init__(self):
        """GPT2 Embeddings Eval."""
        self.model, self.tokenizer = get_gpt_tokenizer()
        self.embeddings = self.model.transformer.wte.weight
        Eval.__init__(self)
        print("Loaded GPT2 model")

    def embedToken(self, word):
        """Different for Word2Vec and GPT2."""
        tmp = torch.mean(self.embeddings[self.tokenizer.encode(word), :], 0).unsqueeze(0).data.cpu().numpy()[0]
        return tmp

    def tokenizeSequence(self, sequence):
        """Different for Word2Vec and GPT2."""
        sequence = self.removeStopWords(sequence)
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
        print("loaded Word2Vec model; time :", time() - tic) 
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
        sequence = self.removeStopWords(sequence)
        toks = simple_preprocess(sequence)
        return self._filter_tokens_not_in_model(toks)




def get_dist_closest(model, text, genre, all_genres):
    """returns lst [top1, top3, genre1, genre2, ..., genre11]"""
    dists = model.distanceToReferenceWords(text) 
    # print(dists)
    rtnlst = []
    # top 1
    res = sorted(dists.items(), key=itemgetter(1), reverse=True)[:3]
    res = [a[0] for a in res]
    # print(res)
    if genre == res[0]:
        rtnlst += [str(1), str(1)] # top1, top2
    elif genre in res[1:]:
        rtnlst += [str(0), str(1)]  # top1, top2
    else:
        rtnlst += [str(0), str(0)]  # top1, top2
    for g in all_genres:
        rtnlst.append(dists[g])
    # print(rtnlst)
    return rtnlst


comedy = ("Comedy", "ARTHUR\nWell, that's just a little thin.\nLIONEL\nYeah, just a little thin maybe.\nArthur sits down next to the desk.\nARTHUR\nIt's a little bit of a joke, but,\nstill there's a lot of funny jokes to the people.")

gpt = GPT2EmbeddingsEval()
# get_dist_closest(gpt, comedy[1], comedy[0]) # test
wv = Word2VecEval()

# Evaluate data/output.csv
df = pd.read_csv('data/output.csv')

# seed_df = pd.DataFrame(referenceWords)
# seed_df.to_csv('data/seed_words_kNN.csv', index=False)
# print("Wrote seedwords to", 'data/seed_words_kNN.csv')

def getCols(emb, gen, genre):
    cols = ['{}_top1_{}'.format(emb, gen), '{}_top3_{}'.format(emb, gen)]
    for g in genre:
        cols.append('{}_{}_{}'.format(emb, g, gen))
    return cols


fname = os.path.join(DATA_DIR, "output_evals_ALL_MODELS_7_GENRES.csv")
with open(fname, "w") as f:
    csv_out = csv.writer(f)
    all_genres = gpt.genre
    cols = ['genre', 'seed_text']
    cols += getCols('w2v', 'GPT', all_genres) 
    cols += getCols('w2v', 'BART', all_genres)
    cols += getCols('w2v', 'bi', all_genres)
    cols += getCols('w2v', 'tri', all_genres)
    cols += getCols('gpt', 'GPT', all_genres)
    cols += getCols('gpt', 'BART', all_genres)
    cols += getCols('gpt', 'bi', all_genres)
    cols += getCols('gpt', 'tri', all_genres)
    print(tuple(cols))
    csv_out.writerow(tuple(cols))

    for i in tqdm(range(len(df))):
        genre = df.iloc[i, 0].replace("<", "").replace(">", "") #
        if genre not in all_genres: 
            print("not evauating genre", genre)
            continue
        seed_text = df.iloc[i, 1] #
        gpt_gen = df.iloc[i, 2]
        bart_gen = df.iloc[i, 3]
        bi = df.iloc[i, 4]
        tri = df.iloc[i, 5]
        generated_texts = [gpt_gen, bart_gen, bi, tri]
        out = [genre, seed_text]
        for texts in generated_texts:
            out += get_dist_closest(wv, texts, genre, all_genres)
        for texts in generated_texts:
            out += get_dist_closest(gpt, texts, genre, all_genres)
        assert len(out) == len(cols)
        csv_out.writerow(tuple(out))



action = ("Action", """wall.\nMAN\nWhat the hell's that?\nINT. PLATFORM - NIGHT\nIn a loud mixture of seats, from the express train. The still sounds as if in the whole plane. Nevertheless, the growing chorus of the passengers impatiently.\nMAN (o.s.)\nHey, I've got a column on that you know. It's just a first\ncheck of the fuel line, Ray. That's it.""")

horror = ("Horror", """wall.\nCAMERA PANS TO A CLOSE SHOT of the man. He starts to walk slowly.\nINSERT - CLOSE ON SIDNEY'S EXPRESSION AREA, which is dropped off of her fingers.\nA HAND plunges INTO FRAME between the clasped side and in the man's hand. He is holding his head. He looks up slowly, slowly.""")

fantasy = ("Fantasy", """and the KNOCKS on the\nground, then turns and runs toward the open crypt. INT. COLD WILLOW'S STUDY - NIGHT\nIn the background, the door to the small rooms of the\nOld Brewery occupies the ancient temple. He walks through the doorway and out into the corridor.""")


