"""
N-gram for controlled text generation for IMSDB movie scripes

NOTE about data:
- some scripts have multiple labels (n-grams doesn't handle that here)
    - idk how it could
"""
import os
import dill as pickle  # to save and load ngram models
from nltk.lm.preprocessing import padded_everygram_pipeline
import pandas as pd
from time import time

# from tqdm import tqdm # tried to use this but don't know how

try:  # Use the default NLTK tokenizer.
    from nltk import word_tokenize, sent_tokenize
    # Testing whether it works.
    # Sometimes it doesn't work on some machines because of setup issues.
    dummy = word_tokenize(sent_tokenize("This is a foobar sentence. Yes it is.")[0])
    print("nltk.word_tokenize works! using this.")
except:  # Use a naive sentence tokenizer and toktok.
    import re
    from nltk.tokenize import ToktokTokenizer
    # See https://stackoverflow.com/a/25736515/610569
    sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)
    # Use the toktok tokenizer that requires no dependencies.
    toktok = ToktokTokenizer()
    word_tokenize = word_tokenize = toktok.tokenize
    print("using nltk.tokenize.ToktokTokenizer for word_tokenize.")

# used for generation
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenize = TreebankWordDetokenizer().detokenize

# Setting up paths
file_loc = os.path.dirname(os.path.abspath(__file__))
HOME_dir = os.path.dirname(os.path.dirname(file_loc))
os.chdir(HOME_dir)

DATA_DIR = os.path.join(HOME_dir, 'data')          # REQ
MODEL_DIR = os.path.join(HOME_dir, 'ngram_models') # REQ
GEN_DIR = os.path.join(HOME_dir, 'ngram_generations')

# Data location
SCRIPTS_ALL = os.path.join(DATA_DIR, 'genres.csv') # REQ: ONLY USING THIS
SCRIPTS_TRAIN = os.path.join(DATA_DIR, 'train.csv')
SCRIPTS_TEST = os.path.join(DATA_DIR, 'test.csv')

exists = lambda x: os.path.exists(x)


# Load raw csv data
for i in (SCRIPTS_ALL, SCRIPTS_TRAIN, SCRIPTS_TEST):
    assert(exists(i)), "Couldn't find data"

# stats
get_stats = lambda table: table.iloc[:, 4:].sum(axis=0)
# print(get_stats(df))
save_stats = lambda table: get_stats(table).to_csv('data/genre_stats.csv', index=True)






# Train model
from nltk.lm import MLE, Lidstone, Laplace, KneserNeyInterpolated, WittenBellInterpolated

import multiprocessing as mp

class ScriptGram(object):
    def __init__(self, n=3, lm="mle", tokenizer=None):
        """Goal: generate controlled text (genre specific)
        Two approaches 
        1. (load_data --> train_models --> generate) 
        2. (load_models --> generate)
        Parameters
        ----------
        n : int
        lm : str ("mle", "lidstone", "laplace", "witten", "kneser")
        word_tokenize : tokenizing function (default nltk.word_tokenize)
        Returns
        ----------
        ScriptGram object 
        For 2. models for given `n` and `lm` must be trained prior
        """
        self.n = n
        self.df = None
        self.all_genres = None
        self.data_dict = {}  # genre : (train_data, padded_sents)
        self.models = {} # genre : nltk.lm.LanguageModel (some LM)
        self.word_tokenizer = word_tokenize if tokenizer is None else tokenizer
        self.lang_models = {
            "mle": MLE,
            "lidstone": Lidstone,
            "laplace": Laplace,
            "witten": WittenBellInterpolated,
            "kneser": KneserNeyInterpolated
        }
        self.lm = lm
        self.LM = self.lang_models[lm]

    def load_models(self):
        """loads models from MODEL_DIR"""
        # get genre names
        gen_stats = os.path.join(DATA_DIR, 'genre_stats.csv')
        gen = pd.read_csv(gen_stats, header=None)
        # print(gen)
        self.all_genres = gen.iloc[:, 0].tolist()
        # print(self.all_genres)
        # get model files
        for genre in self.all_genres:
            modfile = self.model_file_name(genre)
            print("loading...", modfile)
            with open(modfile, 'rb') as fin:
                self.models[genre] = pickle.load(fin)

    def _generate(self, genre, text_seed=None, num_words=1, random_seed=None):
        """generate text wrapper for nltk's model.generate"""
        content = []
        mod = self.models[genre]
        for token in mod.generate(num_words, text_seed=text_seed, random_seed=random_seed):
            if token == '<s>':
                continue
            if token == '</s>':
                break
            content.append(token)
        return detokenize(content)

    def generate_stylized_text(self, genre=None, outFile=None, text_seed=None, num_words=1, random_seed=None):
        """Generate controlled text using trained/saved models
        Params
        -------
        genre : (str, List, None) which genre to generate for
            str - will only generate for given genre
            List - will generate for all given genres 
            None - will generate for ALL genres
        outFile : 
        text_seed :
        num_words : number of words to generate
        random_seed : keep generations consistent (debug purposes)
        """
        gen_lst = []
        if genre is None:
            gen_lst = self.all_genres
        if isinstance(genre, str):
            gen_lst.append(genre)

        columns = ["genre", "text_seed", "generation"]
        outputs = []
        for genre in gen_lst:
            assert genre in self.all_genres, "inappropriate genre entered (confirm with genre_stats.csv for spelling)"
            txt = self._generate(genre, text_seed=text_seed, num_words=num_words, random_seed=random_seed)
            print("------", genre, "------")
            print(txt)
            outputs.append([genre, text_seed, txt])
        if outFile is None:
            outFile = self.generations_file_name()
        outframe = pd.DataFrame(outputs, columns=columns)
        if os.path.exists(outFile):
            outframe.to_csv(outFile, mode='a', index=False, header=False)
        else:
            outframe.to_csv(outFile, mode='w', index=False)

    def generations_file_name(self, lm=None, n=None):
        """All generations will be in one csv."""
        if lm is None: lm = self.lm
        if n is None: n = self.n
        f = '_'.join([lm, str(n) + 'gram']) + ".csv"
        if not os.path.exists(GEN_DIR): os.mkdir(GEN_DIR)
        return os.path.join(GEN_DIR, f)

    def model_file_name(self, genre, lm=None, n=None):
        """Get model name as described by params.
        
        Params
        --------
        genre : str
        lm : str Language Model name (i.e. 'mle') look at __init__
        n : int (probably self.n)
        
        returns
        --------
        fullpath file: str - MODEL_DIR/lm_genre_(n)gram.pkl

        """
        if lm is None: lm = self.lm
        if n is None: n = self.n
        f = '_'.join([lm, genre, str(n) + 'gram']) + ".pkl"
        return os.path.join(MODEL_DIR, f)

    def save_all_models(self):
        for genre in self.all_genres:
            # fname = '_'.join([self.lm, genre, str(self.n)+'gram']) + ".pkl"
            # fname = os.path.join(MODEL_DIR, fname)
            self.save_model(genre)
            # fname = self.model_file_name(genre)
            # if not os.path.exists(MODEL_DIR):
            #     os.mkdir(MODEL_DIR)
            # with open(fname, 'wb') as fout:
            #     pickle.dump(self.models[genre], fout)
            # print("Saved:", fname)

    def _train_wrapper(self, genre):
        """to be used for parallization"""
        # print("training genre:", genre)
        tic = time()
        train, vocab = self.data_dict[genre]
        print("Training genre:", genre)
        self.models[genre].fit(train, vocab)
        print(genre, "training time:", time() - tic)
        return (genre, time() - tic)

    def _train_parallel(self):
        """PARALLEL train doesn't work!"""
        results = []
        on_result = lambda result: results.append(result)
        pool = mp.Pool(processes=mp.cpu_count())
        for genre in self.all_genres:
            print("async training...", genre)
            pool.apply_async(self._train_wrapper, (genre), callback=on_result)
        pool.close()
        pool.join()
        print("Done results:", results)

    def _train_sequential(self):
        """regular non-fancy dumb training"""
        pool = mp.Pool(processes=mp.cpu_count())
        for genre in self.all_genres:
            self._train_wrapper(genre)
            pool.apply_async(self.save_model, (genre,))
            print("saving", genre, "async")
        pool.close()
        pool.join()

    def save_model(self, genre):
        fname = self.model_file_name(genre)
        if not os.path.exists(MODEL_DIR):
            os.mkdir(MODEL_DIR)
        with open(fname, 'wb') as fout:
            pickle.dump(self.models[genre], fout)
        print("Saved:", fname)

    def train_models(self):
        print("--------- Training Model Stats: -----------")
        print("n =", self.n)
        print("lm =", self.lm)
        print("-------------------------------------------")

        # train models MLE for now
        for genre in self.all_genres:
            self.models[genre] = self.LM(self.n)

        # pickle & generators error
        # with mp.Pool(mp.cpu_count()) as pool:
        #     pool.map(self._train_wrapper, self.all_genres)

        tic = time()
        # PARALLEL
        # self._train_parallel()

        # NON PARALLEL
        self._train_sequential()

        # save models to file now
        print("total training + saving time", time() - tic)
        # self.save_all_models()

    def load_data(self):
        """Load & preprocess train data."""
        self.df = pd.read_csv(SCRIPTS_ALL)
        self.all_genres = self.df.columns[4:]
        self.data_dict = {}
        tot = len(self.all_genres)
        tic = time()
        print("----------Preprocessing stats:----------")
        print("n =", self.n)
        print("tokenizer =", self.word_tokenizer)
        print("----------------------------------------")
        for i, genre in enumerate(self.all_genres):
            scripts = self.all_scripts_for_genre(self.df, genre)
            print("processing :", genre, len(scripts), ";", i+1, "of", tot)
            tokenized = self.tokenize_scripts(scripts, genre)
            ngrams, vocab = padded_everygram_pipeline(self.n, tokenized)
            self.data_dict[genre] = (ngrams, vocab)
        print("TOTAL DATA PROCESSING TIME", time() - tic)
        return self.data_dict

    def _parallel_load_genre_to_datadict(self, genre):
        """ DOESN'T WORK """
        scripts = self.all_scripts_for_genre(self.df, genre)
        # print("processing :", genre, len(scripts))
        tokenized = self.tokenize_scripts(scripts, genre)
        ngrams, vocab = padded_everygram_pipeline(self.n, tokenized)
        self.data_dict[genre] = (ngrams, vocab)

    def parallel_load_data(self):
        """load & preprocess train data in parallel
        Won't return anything results will be self.data_dict
        DOESN'T WORK ! 
        """
        self.df = pd.read_csv(SCRIPTS_ALL)
        self.all_genres = self.df.columns[4:]
        self.data_dict = {}
        tot = len(self.all_genres)
        # with tqdm(total=tot) as pbar:
        # on_result = lambda result: pbar.update(result)
        on_result = lambda result: print("done", result)
        pool = mp.Pool(processes=mp.cpu_count())
        for i, genre in enumerate(self.all_genres):
            print("processing :", genre)
            pool.apply_async(self._parallel_load_genre_to_datadict, (genre,), callback=on_result)
        pool.close()
        pool.join()

    def all_scripts_for_genre(self, df, genre):
        """extract all scripts and return list[str]"""
        scripts = df.loc[df[genre] == 1]
        scripts = scripts['script'].tolist()
        return scripts

    def wrap_tokenize(self, s):
        try:
            return list(self.word_tokenizer(s))
        except:
            return []

    def tokenize_scripts(self, scripts, genre):
        """tokenize using nltk
        Returns
        --------
        List[List[String]]
        """
        tokens = []
        tot = len(scripts)
        tic = time()

        for i, s in enumerate(scripts):
            # NOTE: keeping original case of words
            # if i % 10 == 0: print("processing", genre, ";", i, "of", tot, " - time:", time() - tic)
            tokens.append(self.wrap_tokenize(s))
        print("time:", time() - tic)
        # lower case version
        # tokens.append(list(map(str.lower, word_tokenizer(s))))
        return tokens



import sys

if __name__ == "__main__":
    n = int(sys.argv[1])
    """
    ltg = load_data --> train_models --> generate_text
    lg = load_models --> generate_text
    """
    sg = ScriptGram(n = n)
    action = sys.argv[2]
    if action == 'ltg':
        sg.load_data()
        sg.train_models()
        sg.generate_stylized_text(text_seed="The man went to the park", num_words=400)
    elif action == 'lg':
        sg.load_models()
        sg.generate_stylized_text(text_seed="The man went to the park", num_words=400)
    else:
        print("invalid <action> argument:", action)
        exit()







"""
# removing mult col names appropriately
actThril = df.loc[df['Action.Thriller'] == 1]
actThril = actThril.index.values[0]
# print(df.iloc[actThril])
# df.iloc[actThril]['Action'] = 1
df.set_value(actThril, 'Action', 1)
df.set_value(actThril, 'Thriller', 1)
df.drop('Action.Thriller', axis=1, inplace=True)
# print(df.iloc[actThril])

horMys = df.loc[df['Horror.Mystery'] == 1]
horMys = horMys.index.values[0]
# print(df.iloc[horMys])
df.set_value(horMys, 'Horror', 1)
df.set_value(horMys, 'Mystery', 1)
# print(df.iloc[horMys])
df.drop('Horror.Mystery', axis=1, inplace=True)

df.to_csv('data/test_new.csv',index=False)
"""
