## Import libraries

import numpy as np
import tensorflow as tf
import time
import os
from fx_for_chatbot import *

## Importing dataset and data preprocessing
# Import dataset
wd_path = os.getcwd()
folder_name = 'cornell_movie_dialogs_corpus'
file_line = 'movie_lines.txt'
file_conv = 'movie_conversations.txt'
lines = open(os.path.join(wd_path, folder_name, file_line), encoding = 'utf',
             errors = 'ignore').read().split('\n')
convs = open(os.path.join(wd_path, folder_name, file_conv), encoding = 'utf',
             errors = 'ignore').read().split('\n')

# Create dictionary for lines and list for conversations
id2line = {}
conv_list = []
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[-1]
        
for conv in convs[:-1]:
    _conv = conv.split(' +++$+++ ')[-1][2:-2]
    conv_list.append(_conv.split('\', \''))

# Split conversations into questions and answers
questions, answers = [], []
for conv in conv_list:
    for i in range(len(conv) - 1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i + 1]])
        
# Clean text
clean_questions, clean_answers = [], []
for question in questions:
    clean_questions.append(clean_text(question))

for answer in answers:
    clean_answers.append(clean_text(answer))
    
# Create dictionary to map words' occurrences
word_counts = {}
for question in clean_questions:
    for word in question.split():
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1
for answer in clean_answers:
    for word in answer.split():
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1

# Create dictionaries that map question and answer words to unique integer
threshold = 20
questionwords2int = {}
word_id = 0
for word, count in word_counts.items():
    if count >= threshold:
        questionwords2int[word] = word_id
        word_id += 1
answerwords2int = {}
word_id = 0
for word, count in word_counts.items():
    if count >= threshold:
        answerwords2int[word] = word_id
        word_id += 1

# Add last tokens to the dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionwords2int[token] = len(questionwords2int) + 1
    answerwords2int[token] = len(answerwords2int) + 1

# Create inverse dictionary for answerwords2int
answerints2word = {word_int: word 
                   for word, word_int in answerwords2int.items()}

# Add EOS to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Translate all the words to the encoded integers and replace filtered words
encoded_questions, encoded_answers = [], []
for question in clean_questions:
    _question = []
    for word in question.split():
        if word not in questionwords2int: 
            _question.append(questionwords2int['<OUT>'])
        else:
            _question.append(questionwords2int[word])
    encoded_questions.append(_question)
for answer in clean_answers:
    _answer = []
    for word in answer.split():
        if word not in answerwords2int: 
            _answer.append(answerwords2int['<OUT>'])
        else:
            _answer.append(answerwords2int[word])
    encoded_answers.append(_answer)

# Sort questions and answers by the length of the questions
max_length = 25
sorted_questions, sorted_answers = [], []
for length in range(1, max_length + 1):
    for index, question in enumerate(encoded_questions):
        if len(question) == length:
            sorted_questions.append(encoded_questions[index])
            sorted_answers.append(encoded_answers[index])

## Build and train the seq2seq model
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Define session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Load model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Set sequence length
seq_length = tf.placeholder_with_default(25, shape = None,
                                         name = 'sequence_length')

# Get the shape of the inputs tensor
input_shape = tf.shape(input = inputs)

# Get the training and test predictions
train_predictions, test_predictions = seq2seq_model(tf.reverse(tensor = inputs, axis = [-1]),
                                                    targets, keep_prob,
                                                    batch_size, seq_length,
                                                    len(answerwords2int), 
                                                    len(questionwords2int),
                                                    encoding_embedding_size, 
                                                    decoding_embedding_size,
                                                    rnn_size, num_layers,
                                                    questionwords2int)
