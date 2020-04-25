# N-grams, Lexical Evaluation, and Word Visualizations

`ngrams_controlled_gen.py`
> To train and generate n_grams according to genres
- The code was tested with Python 3.6
- Install modules from `requirements.txt` with `pip`
- To initialize n-gram model pass in two arguments
    - `arg1` : number of grams (2, 3, ... (int))
    - `arg2` : action (`ltg` or `lg`)
        - `ltg` = load_data --> train_models --> generate_text
        - `lg` = load_models --> generate_text
- to train bi grams 
    - Download `Datasets/genres.csv` from GoogleDrive linked in main README and place it in `HOME/data/genre.csv`
    - Run: `python ngrams_controlled_gen.py 2 ltg`
- to generate text using saved bigram models
    - train bigram or downlaod ngram model files from GoogelDrive and place it in `HOME/models/ngram_models/`

`embed_evals.py`
> Performs evaluations of generated scripts must be in the following format
- requires `HOME/data/output.csv` in the format (col names may differ):
```
genre,seed_text, gpt_gen, bart_gen, bi_gram_gen, 3_gram_gen
Action, The man.., ..., ..., ..., ...
.
.
.
```
- create eval file : `HOME/data/output_evals_ALL_MODELS_7_GENRES.csv`

`visualize_words.py`
> Used to visualize tfidf and gpt embedding using tSNE plots
- generated files are stored in `HOME/plots/`
- additinal requirements
    - torch
    - pathlib
    - transformers

