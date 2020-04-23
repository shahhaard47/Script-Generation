import pandas as pd

col_list = ["Genre", "Seed_Text", "BERT_cls_gpt_gen", "BERT_cls_2gram", "BERT_cls_3gram"]
df = pd.read_csv("Final generations and eval/BERT_eval.csv", usecols=col_list)

actual_tags = df["Genre"].values
p1_tags = df["BERT_cls_gpt_gen"].values
p2_tags = df["BERT_cls_2gram"].values
p3_tags = df["BERT_cls_3gram"].values

p1_count = 0
p1_total = 0
for i in range(len(actual_tags)):
	if "music" in actual_tags[i]:
		continue
	if actual_tags[i] == p1_tags[i]:
		p1_count += 1
	p1_total += 1
p1_acc = float(p1_count)/p1_total

p2_count = 0
p2_total = 0
for i in range(len(actual_tags)):
	if "music" in actual_tags[i]:
		continue
	if actual_tags[i] == p2_tags[i]:
		p2_count += 1
	p2_total += 1
p2_acc = float(p2_count)/p2_total

p3_count = 0
p3_total = 0
for i in range(len(actual_tags)):
	if "music" in actual_tags[i]:
		continue
	if actual_tags[i] == p3_tags[i]:
		p3_count += 1
	p3_total += 1
p3_acc = float(p3_count)/p3_total

print(f"Accuracies using BERT classifier-- GPT-2: {p1_acc}, 2Gram: {p2_acc}, 3Gram: {p3_acc}")
