# This file includes the main functions used to train the model and verify clusters;

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
	""" function for changing names in the folder
	"""
        for old_name in file_name_list:
                new_name = old_name.replace("r", "t")
                full_old_name = dir + old_name
                full_new_name = dir + new_name
                os.rename(full_old_name, full_new_name)

def load_doc(filename):
	""" load document into memory
	"""
	# open the file as read only
	file = open(filename, 'r', encoding = 'unicode_escape')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def clean_doc(doc):
	""" turn a doc into clean tokens
	"""
	# split into tokens by white space
	doc = doc.replace('-', ' ')
	doc = doc.replace('/', ' ')
	doc = doc.replace('scm', 'supply Chain Management')
	doc = doc.replace('scr', 'supply Chain Resilience') # other texts can be added
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
	tokens = [w for w in tokens if not w in {'the', 'and', 'paper', 'literature',
											 'research', 'study', 'purpose', 'analysi', 'find', 'use'}] # change based on your project
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

def add_doc_to_vocab(filename, vocab):
	""" load .txt file and add to vocab
	"""
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)

def process_docs(directory, vocab):
	""" load all docs in a directory
	"""
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if filename.startswith('sct'): # 'sct' can be changed based on your project
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

def save_list(lines, filename):
	""" save list to file
		"""
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w')
	# write text
	file.write(data)
	# close file
	file.close()

def create_vocabs(cluster_foler_dir, vocab, clusterName):
	""" create vocabs for each cluster
		"""
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

def doc_to_line(filename, vocab):
	""" load abstracts, clean and return line of tokens
		"""
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)

# load all docs in a directory
def process_train_docs(directory, vocab, is_trian):
	""" load all docs in a directory
		"""
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

def prepare_data(train_abs, test_abs, mode):
	""" prepare bag of words encoding of abstracts
	"""
	# create the tokenizer
	tokenizer = Tokenizer()
	# fit the tokenizer on the documents
	tokenizer.fit_on_texts(train_abs)
	# encode training data set
	Xtrain = tokenizer.texts_to_matrix(train_abs, mode=mode)
	# encode training data set
	Xtest = tokenizer.texts_to_matrix(test_abs, mode=mode)
	return Xtrain, Xtest

def evaluate_mode(Xtrain, ytrain, Xtest, ytest):
	""" evaluate the performance of neural network model
		"""
	scores = list()
	n_repeats = 30 # the number of repeats is for checking robust
	n_words = Xtest.shape[1]
	for i in range(n_repeats):
		# define network based on the project
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
		print('%d accuracy is %s' % ((i+1), acc))
	return scores

def predict_sentiment(abstract, vocab, tokenizer, model):
	""" classify a abstract whether belong to this cluster.
		"""
	# clean
	tokens = clean_doc(abstract)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	# convert to line
	line = ' '.join(tokens)
	# encode
	encoded = tokenizer.texts_to_matrix([line], mode='freq')
	# prediction
	yhat = model.predict(encoded, verbose=0)
	return round(yhat[0,0])


