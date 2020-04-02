import numpy as np
from helpers.utils_scripts import ScriptProcessor

import os
import sys
import torch
import random
import argparse


from transformers import GPT2DoubleHeadsModel,GPT2LMHeadModel, GPT2Model
from transformers import GPT2Config
from transformers import GPT2Tokenizer
from helpers.utils_scripts import convert_examples_to_features

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange



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

def select_field(features, field):
	ans = []
	for feature in features:
		for choice in feature.choices_features:
			ans.append(choice[field])
	return ans
	# print([[choice[field] for choice in feature.choices_features] for feature in features])
	# return [[torch.tensor(choice[field]) for choice in feature.choices_features] for feature in features]

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def load_and_cache_examples(args, tokenizer, evaluate=False, test=False):
    #Uncomment for distributed training
    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    task = 'scripts'
    processor = ScriptProcessor()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = "dev"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"
    assert not (evaluate and test)
    
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_len),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        # logger.info("Loading features from cached file %s", cached_features_file)
        print(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        # logger.info("Creating features from dataset file at %s", args.data_dir)
        print(f"Creating features from dataset file at {args.data_dir}")
        # label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif test:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples, genre_dict = processor.get_train_examples(args.data_dir)
        # logger.info("Training number: %s", str(len(examples)))
        print(f"Number of examples: {len(examples)}")
        features = convert_examples_to_features(
            examples,
            genre_dict,
            args.max_seq_len,
            tokenizer,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            model_type = args.model_type
        )
        # if args.local_rank in [-1, 0]:
        # logger.info("Saving features into cached file %s", cached_features_file)
        print(f"Saving features into cached file {cached_features_file}")
        torch.save(features, cached_features_file)

    #Uncomment for distributed training
    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    # print(features)
    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    # all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    # all_cls_ids = torch.tensor(select_field(features, "cls_token_location"), dtype = torch.long)
    # print(all_input_ids[3])
    # print(all_cls_ids[3])
    # sadfsdf

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    return dataset


def train(args, model, tokenizer, train_dataset):
	train_sampler = RandomSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

	if args.max_steps > 0:
	    t_total = args.max_steps
	    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
	else:
	    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
	    {
	        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
	        "weight_decay": args.weight_decay,
	    },
	    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
	]

	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
	scheduler = get_linear_schedule_with_warmup(
	    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
	)

	print("***** Running training *****")
	print(f"  Num examples = {len(train_dataset)}")
	print(f"  Num Epochs = {args.num_train_epochs}" )
	print(
	    f"  Total train batch size (w. accumulation) = {args.train_batch_size * args.gradient_accumulation_steps}"
	)
	print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
	print(f"  Total optimization steps = {t_total}", )

	global_step = 0
	tr_loss, logging_loss = 0.0, 0.0
	best_dev_acc = 0.0
	best_steps = 0
	model.zero_grad()
	train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

	# model_path = './models'
	if not os.path.exists(args.output_dir):
	    os.mkdir(args.output_dir)

	for epoch in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc = "Iteration")
		for step, batch in enumerate(epoch_iterator):
			model.train()
			batch = tuple(t.to(args.device) for t in batch)
			# print(batch)
			inputs = {
				"input_ids": batch[0],
				"attention_mask": batch[1],
				"token_type_ids": batch[2],
				"labels": batch[0]
				}
			outputs = model(**inputs)
			loss = outputs[0]
			# print(loss)
			if args.gradient_accumulation_steps > 1:
				loss = loss / args.gradient_accumulation_steps
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
			tr_loss += loss.item()
			if (step + 1) % args.gradient_accumulation_steps == 0:
				optimizer.step()
				scheduler.step()  # Update learning rate schedule
				model.zero_grad()
				global_step += 1
				if args.logging_steps > 0 and global_step%args.logging_steps == 0:
					print(f"\nAverage loss: {(tr_loss - logging_loss) / args.logging_steps} at global step: {global_step}")
					logging_loss = tr_loss
				if args.save_steps > 0 and global_step % args.save_steps == 0:
					# torch.save(model.state_dict(), os.path.join(model_path, f"{args.model_type}_funnies_{epoch}.pt"))
					output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
					if not os.path.exists(output_dir):
					    os.makedirs(output_dir)
					model_to_save = (
					    model.module if hasattr(model, "module") else model
					)  
					model_to_save.save_pretrained(output_dir)
					tokenizer.save_vocabulary(output_dir)
					torch.save(args, os.path.join(output_dir, "training_args.bin"))
					print(f"Saving model checkpoint to {output_dir}")
			if args.max_steps > 0 and global_step > args.max_steps:
				epoch_iterator.close()
				break
		if args.max_steps > 0 and global_step > args.max_steps:
			train_dataloader.close()
			break
	return global_step, tr_loss / global_step, best_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default='./data',
        type=str,
        help="The input data dir. Should contain the genre.csv file (or other data files) for the task.",
    )
    parser.add_argument(
        "--output_dir",
        default='./models',
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default="gpt2",
        type=str,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="gpt2",
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )

    parser.add_argument(
        "--max_seq_len",
        default = 300,
        type = int,
        help = "maximum length of the input sequence to the transformer"
    )

    parser.add_argument(
        "--train_batch_size",
        default = 12,
        type = int,
        help = "batch size during training"
    )
    parser.add_argument(
        "--eval_batch_size",
        default = 7,
        type = int,
        help = "batch size during training"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--warmup_steps", default=5000, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    
    args = parser.parse_args()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class.from_pretrained(args.model_name_or_path, config = config)

    tokenizer.add_special_tokens({'additional_special_tokens': ['<Comedy>', '<Action>', '<Action.Thriller>', '<Adventure>', '<Animation>', '<Biography>', '<Crime>', '<Drama>', '<Family>', '<Fantasy>', '<Film-Noir>', '<History>', '<Horror>', '<Horror.Mystery>', '<Music>', '<Musical>', '<Romance>', '<Sci-Fi>', '<Short>', '<Sport>', '<Thriller>', '<War>', '<Western>']})
    model.resize_token_embeddings(len(tokenizer)) 
    model.to(device)
    args.device = device

    train_dataset = load_and_cache_examples(args, tokenizer)
    train(args, model, tokenizer, train_dataset)

