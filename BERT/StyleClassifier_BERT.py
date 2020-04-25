"""
FAST-BERT 
"""
# Imports
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import torch
import re
import os
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fast_bert.data_cls import BertDataBunch
from pathlib import Path
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
from torch import Tensor
from fast_bert.prediction import BertClassificationPredictor
from helpers.fast_bert_dataprep import *

def train_fast_bert():
    
    MAX_LEN = 512 # previous model was 300

    text_col = 'script'
    label_col = ['Action', 'Adventure', 'Comedy', 'Crime', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller']
    DATA_PATH = Path('./data/')
    LABEL_PATH = DATA_PATH

    train_file = 'fast_train_' + str(MAX_LEN) + '.csv'
    val_file = 'fast_val_' + str(MAX_LEN) + '.csv'

    goodtogo = check_fastBert_data(MAX_LEN)
    if not goodtogo: die()

    MODEL_NAME = 'bert-base-uncased'

    databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                            tokenizer=MODEL_NAME,
                            train_file=train_file,
                            val_file=val_file,
                            label_file='fast_labels.csv',
                            text_col=text_col,
                            label_col=label_col,
                            batch_size_per_gpu=16,
                            max_seq_length=MAX_LEN,
                            multi_gpu=False,
                            multi_label=True,
                            model_type='bert')


    # **NOTE** remember to change `usePretrained` to True if we've already have a fine-tuned model

    def my_accuracy_thresh(
        y_pred: Tensor,
        y_true: Tensor,
        thresh: float = 0.7,
        sigmoid: bool = False,
    ):
        "Compute accuracy when `y_pred` and `y_true` are the same size."
        if sigmoid:
            y_pred = y_pred.sigmoid()
        return ((y_pred > thresh) == y_true.bool()).float().mean().item()

    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger()
    device_cuda = torch.device("cuda")
    metrics = [{'name': 'accuracy_thresh', 'function': my_accuracy_thresh}]

    OUTPUTDIR = Path('./models/')

    MODEL_PATH = OUTPUTDIR/'model_out_bert_cased'

    usePretrained = False
    if usePretrained:
        pretrained_path = MODEL_PATH
    else:
        pretrained_path='bert-base-uncased'


    # Setting up apex properly on Colab required dowgrading Torch version (check first block of notebook for details)
    learner = BertLearner.from_pretrained_model(
                            databunch,
                            pretrained_path=usePretrained, #MODEL_PATH #(to use saved model)
                            metrics=metrics,
                            device=device_cuda,
                            logger=logger,
                            output_dir=OUTPUTDIR,
                            finetuned_wgts_path=None,
                            warmup_steps=500,
                            multi_gpu=False,
                            is_fp16=False, # need apex setup properly for this (note above)
                            multi_label=True,
                            logging_steps=50)

    learner.fit(epochs=5,
                lr=6e-4,
                validate=True, 	# Evaluate the model after each epoch
                schedule_type="warmup_cosine",
                optimizer_type="lamb")
    # learner.save_model() # no need modified library file to save after every epoch

# simple inference right after training
# texts = ['I really love the Netflix original movies',
		#  'this movie is not worth watching']
# predictions = learner.predict_batch(texts)


# get_ipython().run_line_magic('load_ext', 'tensorboard')
# get_ipython().run_line_magic('tensorboard', '--logdir models/tensorboard/')

# Inference using saved fine-tuned model
def classify_bert(text, model_path):
    """Classify genre using fast-bert.

    Fast-bert automatically uses GPU if `torch.cuda.is_available() == True`

    Parameters
    -----------
    text : <str or list(str)> for single prediction or multiprediction 
    model_path : <str> must contain labels.csv (I've put one in the uploaded version)
            AND all model files (config.json, pytorch_model.bin, special_tokens_map.json, tokenizer_config.json, vocab.txt)

    Returns
    ---------
    str: if type(text) == str
    list: if type(text) == list or numpy array

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = BertClassificationPredictor(
                    model_path=model_path,
                    label_path=model_path, # location for labels.csv file
                    multi_label=True,
                    model_type='bert',
                    do_lower_case=False)
    # predictor.to(device)

    if isinstance(text, str):
        # Single prediction
        pred = predictor.predict(text)
        pred = dict(pred)
        # single_prediction = predictor.predict("just get me result for this text")
    elif isinstance(text, list) or isinstance(text, np.ndarray):
        pred = predictor.predict_batch(text)
        # # Batch predictions
        # texts = [
        #     "this is the first text",
        #     "this is the second text"
        #     ]
        for i in range(len(pred)):
            pred[i] = dict(pred[i])

        # multiple_predictions = predictor.predict_batch(texts)
    else:
        raise ValueError("Unexpected type for input argument `text`")
    return pred

