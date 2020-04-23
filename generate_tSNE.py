from helpers.sample import sample_sequence

import os
import sys
import torch
import random
import argparse
import numpy as np

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
# % matplotlib inline

from tqdm import tqdm, trange
from transformers import GPT2DoubleHeadsModel,GPT2LMHeadModel, GPT2Model
from transformers import GPT2Config
from transformers import GPT2Tokenizer

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    XLNetConfig,
    XLMWithLMHeadModel,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

#Use the right models for language modellnig here 
MODEL_CLASSES = {
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "xlnet": (XLNetConfig, XLMWithLMHeadModel, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
}

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, RobertaConfig, GPT2Config)), ()
)

corpus = api.load('patent-2017')

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, word, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()

def generate_tSNE(args, model, tokenizer):
    embeddings = model.transformer.wte.weight
    w2v = Word2Vec(corpus)
    keys = ['<Comedy>', '<Action>', '<Adventure>', '<Crime>', '<Drama>', '<Fantasy>', '<Horror>', '<Music>', '<Romance>', '<Sci-Fi>', '<Thriller>']
    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embs = []
        words = []
        # print(tokenizer.encode(word))
        print(w2v.most_similar(word.lower().replace("<","").replace(">",""), topn = 10))
        for similar_word,_ in w2v.most_similar(word.lower().replace("<","").replace(">",""), topn = 10):
            words.append(similar_word)
            embs.append(torch.mean(embeddings[tokenizer.encode(similar_word), :],0).unsqueeze(0).data.cpu().numpy())
        embs.append(torch.mean(embeddings[tokenizer.encode(word), :],0).unsqueeze(0).data.cpu().numpy())
        embedding_clusters.append(embs)
        word_clusters.append(words)
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    print(embeddings_clusters.shape)
    tsne_model_en_2d = TSNE(perplexity=5, n_components=2, init='pca', n_iter=1000, random_state=2)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

    tsne_plot_similar_words('Genre embeddings plot', keys, embeddings_en_2d, word_clusters, 0.7,
                        './plots/genre_vis/similar_genres.png')

def tsne_plot(model, tokenizer):
    "Creates and TSNE model and plots it"
    embeddings = model.transformer.wte.weight
    keys = ['<Comedy>', '<Action>', '<Adventure>', '<Crime>', '<Drama>', '<Fantasy>', '<Horror>', '<Music>', '<Romance>', '<Sci-Fi>', '<Thriller>']
    labels = []
    tokens = []

    for word in keys:
        tokens.append(embeddings[tokenizer.encode(word)])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default='./models',
        type=str,
        help="path to the checkpoint file you want to use",
    )
    parser.add_argument(
        "--model_type",
        default="gpt2",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2",
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    args = parser.parse_args()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class.from_pretrained(args.model_name_or_path, config = config)

    # tokenizer.add_special_tokens({'additional_special_tokens': ['<Comedy>', '<Action>', '<Action.Thriller>', '<Adventure>', '<Animation>', '<Biography>', '<Crime>', '<Drama>', '<Family>', '<Fantasy>', '<Film-Noir>', '<History>', '<Horror>', '<Horror.Mystery>', '<Music>', '<Musical>', '<Romance>', '<Sci-Fi>', '<Short>', '<Sport>', '<Thriller>', '<War>', '<Western>']})
    tokenizer.add_special_tokens({'additional_special_tokens': ['<Comedy>', '<Action>', '<Adventure>', '<Crime>', '<Drama>', '<Fantasy>', '<Horror>', '<Music>', '<Romance>', '<Sci-Fi>', '<Thriller>']})
    model.resize_token_embeddings(len(tokenizer)) 
    model.to(device)
    args.device = device

    checkpoints = [args.checkpoint]
    print(f"Evaluate the following checkpoints: {checkpoints}")
    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1] if len(checkpoint.split("-")) > 1 else ""
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        # print(model.transformer.wte.weight[0][:100])
        model = model_class.from_pretrained(checkpoint)
        # print(model.transformer.wte.weight[0][:100])
        # sdfsdf
        model.to(args.device)
        # tsne_plot(model, tokenizer)
        generate_tSNE(args, model, tokenizer)
        # result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        # results.update(result)