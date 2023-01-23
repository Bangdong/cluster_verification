import os, shutil, random

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
from keras.layers import Dropout
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot

from mainFunctions import load_doc, clean_doc, add_doc_to_vocab, doc_to_line, process_docs, prepare_data, save_list, change_file_name
from mainFunctions import process_train_docs
from mainFunctions import evaluate_mode

source_list = ['1_network_design', '2_decision_system', '3_supply_mgmt', '4_chain_coordination',
               '5_behavior_chain', '6_buiding_SCR', '7_framework_design', '8_theory', '9_risk_mitigation',
               '10_implication', '11_factors']

for j in range(11):

    #################### contain the 1 content ##################
    testedCluster = source_list[j]

    os.makedirs('data/results', exist_ok=True)

    edf = pd.DataFrame(index=range(1), columns=range(4))
    edf.columns = ['binary', 'count', 'tfidf', 'freq']
    edf.to_csv("E:/pythonProject/scrCovid/data/results/" + testedCluster + "out.csv",
               index=False)  # new results for one sample

    for i in range(10): # random sampling abstracts to check robust

        results = DataFrame()
        # name for all folders

        # path to source directory
        src_dir = 'E:/pythonProject/scrCovid/data/' + testedCluster
        num_dir_t = len(os.listdir(src_dir))

        # path to destination directory
        dest_dir = 'E:/pythonProject/scrCovid/data/temp'

        # check whether there exists such a file
        if os.path.isdir(dest_dir):
            shutil.rmtree(dest_dir)
            os.mkdir(dest_dir)
        else:
            os.mkdir(dest_dir)

        # getting all the files in the source directory
        files = os.listdir(src_dir)
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)

        #################### contain the abstracts that not belong to tested cluster ##################

        # path to destination directory
        file_dir_nt = 'E:/pythonProject/scrCovid/data/temp_0'

        # check whether there exists such a file
        if os.path.isdir(file_dir_nt):
            shutil.rmtree(file_dir_nt)
            os.mkdir(file_dir_nt)
        else:
            os.mkdir(file_dir_nt)

        counter = 0
        for clusterName in source_list:
            if clusterName == testedCluster:
                continue
            else:
                name_dir_nt = 'E:/pythonProject/scrCovid/data/' + clusterName
                for filename in os.listdir(name_dir_nt):
                    name_source_nt = name_dir_nt + '/' + filename
                    # print(name_source_nt)
                    counter += 1
                    name_des_nt = 'E:/pythonProject/scrCovid/data/temp_0/' + 'scr' + str(counter)
                    shutil.copyfile(name_source_nt, name_des_nt)

        random_file = random.sample(os.listdir('E:/pythonProject/scrCovid/data/temp_0/'), num_dir_t)

        # check whether there exisits such a file to contain the random un targeted sample;
        random_file_dir_nt = 'E:/pythonProject/scrCovid/data/temp_1'
        if os.path.isdir(random_file_dir_nt):
            shutil.rmtree(random_file_dir_nt)
            os.mkdir(random_file_dir_nt)
        else:
            os.mkdir(random_file_dir_nt)

        for random_nt in random_file:
            random_dir_nt = 'E:/pythonProject/scrCovid/data/temp_0/' + random_nt
            random_dir_nt_t1 = 'E:/pythonProject/scrCovid/data/temp_1/' + random_nt  # save the random abstracts in temp_1
            shutil.copyfile(random_dir_nt, random_dir_nt_t1)

        # sort out the last 20% tested sample and change the name
        # get the list
        name_file_t = sorted(os.listdir('E:/pythonProject/scrCovid/data/temp/'), key=len)
        name_file_nt = sorted(os.listdir('E:/pythonProject/scrCovid/data/temp_1/'), key=len)
        number_for_test = int(len(name_file_t) * 0.2)
        number_for_train = len(name_file_t) - number_for_test

        name_file_test_t = name_file_t[-number_for_test:]
        name_file_test_nt = name_file_nt[-number_for_test:]

        # change the last 20% names in the list
        change_file_name(name_file_test_t, 'E:/pythonProject/scrCovid/data/temp/')
        change_file_name(name_file_test_nt, 'E:/pythonProject/scrCovid/data/temp_1/')

        # define vocab
        vocab = Counter()
        # add all docs to vocab
        process_docs('E:/pythonProject/scrCovid/data/temp', vocab)
        process_docs('E:/pythonProject/scrCovid/data/temp_1', vocab)
        # print the size of the vocab
        print(len(vocab))
        # print the top words in the vocab
        print(vocab.most_common(50))

        # keep tokens with a min occurrence
        min_occurane = 1
        tokens = [k for k, c in vocab.items() if c >= min_occurane]
        print(len(tokens))

        # save tokens to a vocabulary file
        save_list(tokens, 'E:/pythonProject/scrCovid/data/vocab_temp.txt')

        # load the vocabulary
        vocab_filename = 'E:/pythonProject/scrCovid/data/vocab_temp.txt'
        vocab = load_doc(vocab_filename)
        vocab = vocab.split()
        vocab = set(vocab)

        # load all training reviews
        positive_lines = process_train_docs('data/temp', vocab, True)
        negative_lines = process_train_docs('data/temp_1', vocab, True)
        train_docs = positive_lines + negative_lines

        # load all test reviews
        positive_lines_test = process_train_docs('data/temp', vocab, False)
        negative_lines_test = process_train_docs('data/temp_1', vocab, False)
        test_docs = positive_lines_test + negative_lines_test

        # prepare labels
        ytrain = array([1 for _ in range(number_for_train)] + [0 for _ in range(number_for_train)])
        ytest = array([1 for _ in range(number_for_test)] + [0 for _ in range(number_for_test)])

        modes = ['binary', 'count', 'tfidf', 'freq'] # use four approaches to check the robust
        for mode in modes:
            # prepare data for mode
            Xtrain, Xtest = prepare_data(train_docs, test_docs, mode)
            # evaluate model on data for mode
            results[mode] = evaluate_mode(Xtrain, ytrain, Xtest, ytest)
        results.to_csv("E:/pythonProject/scrCovid/data/results/" + testedCluster + "out0.csv",
                       index=False)  # new results for one sample

        df = pd.read_csv("E:/pythonProject/scrCovid/data/results/" + testedCluster + "out0.csv")
        # df.drop(columns=df.columns[0], axis=1, inplace=True)

        df1 = pd.read_csv("E:/pythonProject/scrCovid/data/results/" + testedCluster + "out.csv")
        # df1.drop(columns=df.columns[0], axis=1, inplace=True)

        summary = pd.concat([df, df1], ignore_index=True)
        summary.to_csv("E:/pythonProject/scrCovid/data/results/" + testedCluster + "out.csv", index=False)
        os.remove(
            "E:/pythonProject/scrCovid/data/results/" + testedCluster + "out0.csv")  # remove the temp file once finish the update



