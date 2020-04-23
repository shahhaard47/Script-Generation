from sklearn.metrics import classification_report
import pandas as pd
from StyleClassifier_BERT import *
from tqdm import tqdm, trange
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


col_list = ["Genre", "Seed Text", "Generated Script"]
my_genres = ["<Comedy>", "<Action>", "<Horror>", "<Romance>", "<Sci-Fi>", "<Fantasy>", "<Thriller>"]
df = pd.read_csv("data/output_270000.csv", usecols=col_list)
bert_path = "./models/model_out_bert_cased"

y_test = df["Genre"].values
scripts = df["Generated Script"].values
y_pred = []
all_stopwords = stopwords.words('english')
all_stopwords.extend(["i\'m", "i\'ve", "we\'ve", "we\'ll", "we\'re"])
train_iterator = trange(int(len(y_test)), desc="examples")
inp_scripts = []
for i in train_iterator:
	#Comment/uncomment one of the next two lines to toggle removal of stopwords
	# script_without_sw = ' '.join([word for word in scripts[i].lower().split() if not word in all_stopwords])
	script_without_sw = scripts[i]
	inp_scripts.append(script_without_sw)

preds = classify_bert(inp_scripts, bert_path)
for pred in preds:
	sorted_pred = sorted(pred.items(), key=lambda x: x[1], reverse = True)
	j = 0
	while '<' + sorted_pred[j][0] + '>' not in my_genres:
		j+=1
	# print(y_test[i])
	# print(sorted_pred[j][0])
	y_pred.append('<' + sorted_pred[j][0] + '>')
print(classification_report(y_test, y_pred,target_names=my_genres))