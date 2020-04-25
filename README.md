## **Pytorch-Genre-Based-Script-Generation**


## Fine-Tune on Movie Script dataset

```
python train_scripts.py
```
Note: Edit argument parameters if you want to use non-default hyper-parameters. 


## Generate Style-based scripts using trained model

1. The following command fine-tunes the chosen model on SWAG. The available models are GPT-2. (BERT and RoBERTa should work but I haven't tested them yet)
```
python generate_scripts.py \
--text "text prompt to initialize generation" \
--genres "<Genre1> <Genre2> <Genre3> ....." \
--checkpoint './models/<checkpoint_folder>'
--learning_rate 3e-4 
```

Note: For initializing genres make sure you follow the same format as shown above, ex: "<Comedy> <Action>"

## Data

Our movie scripts dataset, model files, intermediate pickle files, and generated examples can be downloaded here:
- [Controlled Movie Scripts Generation Data](https://drive.google.com/open?id=1r5nx1iXkjWsjXx9qHjz6Lr7LDnwNzMlB&authuser=hks32@njit.edu&usp=drive_fs)

### Drive folder descriptions

`./Datasets/` 
- includes raw and preprocessed IMSDB data
- `fast_*` files contain the chunked movie scripts to (300 and 512 token chunks)
    - They are in the format required by *FastBert* library
    - They are useful when running `StyleClassifier_BERT.py` or `notebooks/StyleClassifier_BERT.ipynb`
Download genre.csv file and store it in ./data/
- `IMSDB_movies_dataset.csv` contains ALL the movies data
- If downloaded should be stored in **`HOME/data/`** (HOME refers to project home wherever the repo is cloned)

`./Models/`
- Contain model files for GPT2, BART, BERT, and N-Grams
- Also contain other misc pickle files for GPT and tfidf word vectors tSNE visualizations
- If downloaded should be stored in **`HOME/models/<ngram_models>` 
    - if downloading ngram models, add add `ngram_models` to the path

`./Generations/`
- generated text is can be found in this repo (`HOME/data/`) but can also be donwloaded from gDrive.

`./EvaluationStatistics/`
- These are also available here (`HOME/data`) but can also be downloaded from gDrive/

## License

- OpenAi/GPT2 follow MIT license, huggingface/pytorch-pretrained-BERT is Apache license. 
- We follow MIT license with original GPT2 repository

