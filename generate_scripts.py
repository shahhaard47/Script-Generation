from helpers.sample import sample_sequence

import os
import sys
import torch
import random
import argparse
import numpy as np

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

def generate(args, model, tokenizer):

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model.eval()

    if args.length == -1:
        args.length = 200
    elif args.length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)


    # print(args.genres + " " + args.text)
    context_tokens = tokenizer.encode(args.genres + " " + args.text)
    # print(context_tokens)
    # sdfsdf
    
    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        out = sample_sequence(
            model=model, length=args.length,
            context=context_tokens  if not  args.unconditional else None,
            start_token=tokenizer.encoder['<|endoftext|>'] if args.unconditional else None,
            batch_size=args.batch_size,
            sample = False,
            temperature=args.temperature, top_k=args.top_k, device=device, vocab = tokenizer.decoder
        )
        out = out[:, len(context_tokens):].tolist()
        # print(out)
        for i in range(args.batch_size):
            generated += 1
            text = tokenizer.decode(out[i])
            if args.quiet is False:
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)

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
    parser.add_argument(
        "--batch_size",
        default = -1,
        type = int,
        help = "batch size during training"
    )
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--genres", type=str, required=True)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--quiet", type=bool, default=False)
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
        generate(args, model, tokenizer)
        # result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        # results.update(result)