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


this_file = os.path.dirname(os.path.abspath(__file__))
keys = ['<Comedy>', '<Action>', '<Adventure>', '<Crime>', '<Drama>', '<Fantasy>', '<Horror>', '<Music>', '<Romance>', 'Sci-Fi', '<Thriller>']
referenceWords = {
    "Action": 'military,battle,shoot,punch,resistance,war,conflict,execute,prosecution,force'.split(','),
    "Comedy": 'humor,clowning,funny,laughter,joke,sarcasm,goofy,playful,hilarious,silly'.split(','),
    "Adventure": 'risk,hazard,chance,journey,voyage,venture,epic,exploring,superhero,adventurers'.split(','),
    "Crime": 'theft,murder,felony,rape,fraud,victim,police,corrupt,kill,violence'.split(','),
    "Drama": 'narrative,emotional,depth,relationship,issues,feelings,grief,culture,tension,love,theatrical'.split(','),
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

    # def embedSequence(self):
    #     pass

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

    def distanceToReferenceWords(self, sequence, genre=None, useSum=False):
        """Gets distance between mean/sum of each category and mean/sum of sequence.
        if genre is None: output list of (genre, prob) with genre of lowest distance first
        if genre is str: return (genre, prob) tuple
        """
        if isinstance(sequence, str):
            # tokenize sequence
            sequence = self.tokenizeSequence(sequence)
        probs = []
        if not useSum: 
            # compare means
            inp_seq = self.averageTokenEmbeds(sequence)
            refWords = self.refWordsMeans
            # print(inp_seq.shape)
            # print(self.refWordsMeans.shape)
        else:
            # compare sums
            inp_seq = self.sumTokenEmbeds(sequence)
            refWords = self.refWordsSums
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
            probs.append((genre, self.getDist(inp_seq, refWords[i])))
            # print(probs[-1])
        probs.sort(key = operator.itemgetter(1))
        # for p in probs:
        #     print(p)
        return probs



wv = Word2VecEval()

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

# print(tst_gen)

# model, tokenizer = get_gpt_tokenizer()

# tokens = tokenizer.encode(tst_gen[1])
# # print(tokens)

# for t in tokens:
#     # print(t, tokenizer.decode(t))
#     pass

# for word in referenceWords['Action']:
#     print(model[word])


