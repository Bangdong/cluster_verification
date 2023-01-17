
import os, random
from pathlib import Path
from numpy import array

from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from pattern.text.en import singularize

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras.layers import Dense

def change_file_name(file_name_list, dir):
        for old_name in file_name_list:
                new_name = old_name.replace("r", "t")
                full_old_name = dir + old_name
                full_new_name = dir + new_name
                os.rename(full_old_name, full_new_name)

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding = 'unicode_escape')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	doc = doc.replace('-', ' ')
	doc = doc.replace('/', ' ')
	doc = doc.replace('supply chain', 'supplychain')
	doc = doc.replace('supply chain management', 'supplyChainManagement')
	doc = doc.replace('scr', 'supplyChainResilience')
	doc = doc.replace('SCR', 'supplyChainResilience')
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# convert to lower case
	tokens = [word.lower() for word in tokens]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# simple sense for verb
	tokens = [WordNetLemmatizer().lemmatize(word,'v') for word in tokens]
	# convert plural to singular
	tokens = [singularize(word) for word in tokens]
	# remove meaningless words
	tokens = [w for w in tokens if not w in {'the', 'and', 'paper', 'literature', 'research', 'study', 'purpose', 'analysi', 'find', 'use'}]
	# remove interrupting words
	# tokens = [w for w in tokens if not w in {'sc', 'supply', 'chain'}]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if filename.startswith('sct'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# save list to file
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()

# create vocabs for each cluster
def create_vocabs(cluster_foler_dir, vocab, clusterName):
	# add all docs to vocab
	process_docs(cluster_foler_dir, vocab)
	# print the size of the vocab
	# print(len(vocab))
	# print the top words in the vocab
	print(vocab.most_common(50))
	# keep tokens with a min occurrence
	min_occurane = 1
	tokens = [k for k, c in vocab.items() if c >= min_occurane]
	tokens_fre = [[k, c] for k, c in vocab.items() if c >= min_occurane]
	print(len(tokens_fre))
	# Color = ['Default' for i in range(1279)]

	# clusterName_df = pd.DataFrame(tokens_fre, columns=['token', 'size'])
	# clusterName_df.to_csv("E:/pythonProject/scrCovid/data/clusterVocabs/" + clusterName + ".csv")

	#print(len(tokens))

	# save tokens to a vocabulary file
	save_list(tokens, "E:/pythonProject/scrCovid/data/clusterVocabs/" + clusterName + ".txt")

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)

# load all docs in a directory
def process_train_docs(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('sct'):
			continue
		if not is_trian and not filename.startswith('sct'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines


# prepare bag of words encoding of docs
def prepare_data(train_docs, test_docs, mode):
	# create the tokenizer
	tokenizer = Tokenizer()
	# fit the tokenizer on the documents
	tokenizer.fit_on_texts(train_docs)
	# encode training data set
	Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
	# encode training data set
	Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
	return Xtrain, Xtest


# evaluate a neural network model
def evaluate_mode(Xtrain, ytrain, Xtest, ytest):
	scores = list()
	n_repeats = 30
	n_words = Xtest.shape[1]
	for i in range(n_repeats):
		# define network
		model = Sequential()
		model.add(Dense(50, input_shape=(n_words,), activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		# compile network
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		# fit network
		model.fit(Xtrain, ytrain, epochs=50, verbose=2)
		# evaluate
		loss, acc = model.evaluate(Xtest, ytest, verbose=0)
		scores.append(acc)
		print('%d accuracy: %s' % ((i+1), acc))
	return scores

# classify a abstract whether belong to this cluster.
def predict_sentiment(review, vocab, tokenizer, model):
	# clean
	tokens = clean_doc(review)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	# convert to line
	line = ' '.join(tokens)
	# encode
	encoded = tokenizer.texts_to_matrix([line], mode='freq')
	# prediction
	yhat = model.predict(encoded, verbose=0)
	return round(yhat[0,0])


