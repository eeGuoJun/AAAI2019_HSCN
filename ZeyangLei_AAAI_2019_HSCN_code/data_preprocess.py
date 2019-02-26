import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\-\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polarity.pos").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polarity.neg").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]
def load_data_and_labels1():
    """
    Loads SST polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    train_examples = list(open("./data/train.txt").readlines())
    train_examples = [s.strip() for s in train_examples]
    dev_examples = list(open("./data/dev.txt").readlines())
    dev_examples = [s.strip() for s in  dev_examples]
    test_examples = list(open("./data/test.txt").readlines())
    test_examples = [s.strip() for s in  test_examples]
    # Split by words
    x_text = train_examples +dev_examples+test_examples
    x_text1 = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ")[1:] for s in x_text1]
    # Generate labels
    y=[]
    """
    for s in x_text1:
		label=s.split(" ")[0]
		if label=='0':
			y.append([1,0,0,0,0])
		elif label=='1':
			y.append([0,1,0,0,0])
		elif label=='2':
			y.append([0,0,1,0,0])
		elif label=='3':
			y.append([0,0,0,1,0])
		else:
			y.append([0,0,0,0,1])
    """
    for s in x_text1:
		label=s.split(" ")[0]
		if label=='0':
			y.append([0.9,0.02,0.02,0.02,0.02])
		elif label=='1':
			y.append([0.02,0.9,0.02,0.02,0.02])
		elif label=='2':
			y.append([0.02,0.02,0.9,0.02,0.02])
		elif label=='3':
			y.append([0.02,0.02,0.02,0.9,0.02])
		else:
			y.append([0.02,0.02,0.02,0.02,0.9])
    y=np.array(y)
    return [x_text, y]
def load_data_and_labels2():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polarity.pos").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polarity.neg").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    positive_examples_sent = list(open("./data/rt-polarity.pos_sent.txt").readlines())
    positive_examples_sent = [s.strip() for s in positive_examples_sent]
    negative_examples_sent = list(open("./data/rt-polarity.neg_sent.txt").readlines())
    negative_examples_sent = [s.strip() for s in negative_examples_sent]	
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split() for s in x_text]
    x_text_sent = positive_examples_sent + negative_examples_sent
    x_text_sent = [clean_str(sent) for sent in x_text_sent]
    x_text_sent = [s.split() for s in x_text_sent]	
    # Generate labels
    positive_labels = [[0.02, 0.98] for _ in positive_examples]
    negative_labels = [[0.98, 0.02] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text,x_text_sent,y]
def load_data_and_labels3():
    """
    Loads SST polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    train_examples = list(open("./data/train.txt").readlines())
    train_examples = [s.strip() for s in train_examples]
    dev_examples = list(open("./data/dev.txt").readlines())
    dev_examples = [s.strip() for s in  dev_examples]
    test_examples = list(open("./data/test.txt").readlines())
    test_examples = [s.strip() for s in  test_examples]
    train_examples_sent = list(open("./data/train_sent.txt").readlines())
    train_examples_sent = [s.strip() for s in train_examples_sent]
    dev_examples_sent = list(open("./data/dev_sent.txt").readlines())
    dev_examples_sent = [s.strip() for s in  dev_examples_sent]
    test_examples_sent = list(open("./data/test_sent.txt").readlines())
    test_examples_sent = [s.strip() for s in  test_examples_sent]
    # Split by words
    x_text = train_examples +dev_examples+test_examples
    x_text1 = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ")[1:] for s in x_text1]
    x_text_sent = train_examples_sent + dev_examples_sent + test_examples_sent
    x_text_sent = [clean_str(sent) for sent in x_text_sent]
    x_text_sent = [s.split() for s in x_text_sent]	
    # Generate labels
    y=[]
    """
    for s in x_text1:
		label=s.split(" ")[0]
		if label=='0':
			y.append([1,0,0,0,0])
		elif label=='1':
			y.append([0,1,0,0,0])
		elif label=='2':
			y.append([0,0,1,0,0])
		elif label=='3':
			y.append([0,0,0,1,0])
		else:
			y.append([0,0,0,0,1])
    """
    for s in x_text1:
		label=s.split(" ")[0]
		if label=='0':
			y.append([0.9,0.02,0.02,0.02,0.02])
		elif label=='1':
			y.append([0.02,0.9,0.02,0.02,0.02])
		elif label=='2':
			y.append([0.02,0.02,0.9,0.02,0.02])
		elif label=='3':
			y.append([0.02,0.02,0.02,0.9,0.02])
		else:
			y.append([0.02,0.02,0.02,0.02,0.9])
    y=np.array(y)
    return [x_text,x_text_sent,y]
def load_data_and_labels_MR():
    """
    Loads SST polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    train_examples = list(open("./data/train_MR.txt").readlines())
    train_examples = [s.strip() for s in train_examples]
    dev_examples = list(open("./data/dev_MR.txt").readlines())
    dev_examples = [s.strip() for s in  dev_examples]
    test_examples = list(open("./data/test_MR.txt").readlines())
    test_examples = [s.strip() for s in  test_examples]
    train_examples_sent = list(open("./data/train_sent_MR.txt").readlines())
    train_examples_sent = [s.strip() for s in train_examples_sent]
    dev_examples_sent = list(open("./data/dev_sent_MR.txt").readlines())
    dev_examples_sent = [s.strip() for s in  dev_examples_sent]
    test_examples_sent = list(open("./data/test_sent_MR.txt").readlines())
    test_examples_sent = [s.strip() for s in  test_examples_sent]
    # Split by words
    x_text = train_examples +dev_examples+test_examples
    x_text1 = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ")[1:] for s in x_text1]
    x_text_sent = train_examples_sent + dev_examples_sent + test_examples_sent
    x_text_sent = [clean_str(sent) for sent in x_text_sent]
    x_text_sent = [s.split() for s in x_text_sent]	
    # Generate labels
    y=[]
    for s in x_text1:
		label=s.split(" ")[0]
		if label=='0':
			y.append(0.02)
		else:
			y.append(0.98)
    y=np.array(y)
    return [x_text,x_text_sent,y]
def load_data_and_labels_aspect(dataset):
    """
    Loads Aspect polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    train_examples = list(open("./dataset/"+dataset+"/train.txt").readlines())
    train_examples = [s.strip() for s in train_examples]
    test_examples = list(open("./dataset/"+dataset+"/test.txt").readlines())
    test_examples = [s.strip() for s in  test_examples]
    train_examples_aspect = list(open("./dataset/"+dataset+"/train_aspect.txt").readlines())
    train_examples_aspect = [s.strip() for s in train_examples_aspect]
    test_examples_aspect = list(open("./dataset/"+dataset+"/test_aspect.txt").readlines())
    test_examples_aspect = [s.strip() for s in  test_examples_aspect]
    # Split by words
    x_text = train_examples+test_examples
    x_text1 = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ")[1:] for s in x_text1]
    x_text_aspect = train_examples_aspect+test_examples_aspect
    x_text_aspect = [clean_str(sent) for sent in x_text_aspect]
    x_text_aspect = [s.split() for s in x_text_aspect]
    # Generate labels
    y=[]
    for s in x_text1:
		label=int(s.split(" ")[0])
		# print label
		if label==(-1):
			y.append([1,0,0])
		elif label==0:
			y.append([0,1,0])
		else:
			y.append([0,0,1])
    y=np.array(y)
    # print y
    return [x_text,x_text_aspect,y]
def load_data_and_labels_aspect_QA(dataset):
    """
    Loads Aspect polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    train_examples = list(open("./dataset/"+dataset+"/train.txt").readlines())
    train_examples = [s.strip() for s in train_examples]
    dev_examples = list(open("./dataset/"+dataset+"/dev.txt").readlines())
    dev_examples = [s.strip() for s in dev_examples]
    test_examples = list(open("./dataset/"+dataset+"/test.txt").readlines())
    test_examples = [s.strip() for s in  test_examples]
    train_examples_aspect = list(open("./dataset/"+dataset+"/train_aspect.txt").readlines())
    train_examples_aspect = [s.strip() for s in train_examples_aspect]
    dev_examples_aspect = list(open("./dataset/"+dataset+"/dev_aspect.txt").readlines())
    dev_examples_aspect = [s.strip() for s in dev_examples_aspect]
    test_examples_aspect = list(open("./dataset/"+dataset+"/test_aspect.txt").readlines())
    test_examples_aspect = [s.strip() for s in  test_examples_aspect]
    # Split by words
    x_text = train_examples+dev_examples+test_examples
    x_text1 = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ")[1:] for s in x_text1]
    x_text_aspect = train_examples_aspect+test_examples_aspect+dev_examples_aspect
    x_text_aspect = [clean_str(sent) for sent in x_text_aspect]
    x_text_aspect = [s.split() for s in x_text_aspect]
    # Generate labels
    y=[]
    for s in x_text1:
		label=s.split(" ")[0]
		if label=='0':
			y.append([1,0])
			# print '0'
		else:
			y.append([0,1])
			# print '1'
    y=np.array(y)
    print y
    return [x_text,x_text_aspect,y]
def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

def load_data1():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels1()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]
def load_data2():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences,sentences_sent,labels = load_data_and_labels2()
    sentences_padded = pad_sentences(sentences)
    sentences_sent_padded=pad_sentences(sentences_sent)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    x_sent,_=build_input_data(sentences_sent_padded, labels, vocabulary)
    return [x, x_sent,y, vocabulary, vocabulary_inv]
def load_data3():
    """
    Loads and preprocessed data for the SST dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences,sentences_sent,labels = load_data_and_labels3()
    sentences_padded = pad_sentences(sentences)
    sentences_sent_padded=pad_sentences(sentences_sent)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    x_sent,_=build_input_data(sentences_sent_padded, labels, vocabulary)
    return [x, x_sent,y, vocabulary, vocabulary_inv]
def load_data_MR():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences,sentences_sent,labels = load_data_and_labels_MR()
    sentences_padded = pad_sentences(sentences)
    sentences_sent_padded=pad_sentences(sentences_sent)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    x_sent,_=build_input_data(sentences_sent_padded, labels, vocabulary)
    return [x, x_sent,y, vocabulary, vocabulary_inv]
def load_data_aspect(dataset):
    """
    Loads and preprocessed data for the Aspect dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences,sentences_aspect,labels = load_data_and_labels_aspect(dataset)
    sentences_padded = pad_sentences(sentences)
    sentences_aspect_padded=pad_sentences(sentences_aspect)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    x_aspect,_=build_input_data(sentences_aspect_padded, labels, vocabulary)
    return [x, x_aspect,y, vocabulary, vocabulary_inv]
def load_data_aspect_QA(dataset):
    """
    Loads and preprocessed data for the Aspect dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences,sentences_aspect,labels = load_data_and_labels_aspect_QA(dataset)
    sentences_padded = pad_sentences(sentences)
    sentences_aspect_padded=pad_sentences(sentences_aspect)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded+sentences_aspect_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    x_aspect,_=build_input_data(sentences_aspect_padded, labels, vocabulary)
    return [x, x_aspect,y, vocabulary, vocabulary_inv]
def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
