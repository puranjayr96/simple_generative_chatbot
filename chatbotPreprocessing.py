import os
import json
import nltk
import gensim
import numpy as np
from gensim import corpora, models, similarities
import pickle
from keras.models import Model
from keras.layers import Input, LSTM, Dense,Bidirectional,Embedding,Concatenate

# Pre-trained model for word2vec
model = models.Word2Vec.load("word2vec.bin")
# Conversation file
file = open("conversation.json")
data = json.load(file)
cor = data["conversations"]
x = []
y = []

# Seperating each line into inputs and targets
for i in range(len(cor)):
    for j in range(len(cor[i])):
        if j<len(cor[i]) - 1 :
            x.append(cor[i][j])
            y.append(cor[i][j+1])

# Seperation of each word (Tokenization)
tok_x = []
tok_y = []
for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))

# To mark the start of a sentence
sentstart = np.zeros((300,),dtype=np.float32)
# To mark the end of a sentence
sentend = np.ones((300,),dtype=np.float32)

# Vectorize the sentences
vec_x = []
for sent in tok_x:
    sent_vec = [model[w] for w in sent if w in model.vocab]
    vec_x.append(sent_vec)

vec_y = []
for sent in tok_y:
    sent_vec = [model[w] for w in sent if w in model.vocab]
    vec_y.append(sent_vec)

for tok_sent in vec_x:
    tok_sent[14:] = []
    tok_sent.append(sentend)

for tok_sent in vec_x:
    if len(tok_sent) < 15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)

for tok_sent in vec_y:
    tok_sent[14:] = []
    tok_sent.append(sentend)

for tok_sent in vec_y:
    if len(tok_sent) < 15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)

with open('conversation.pkl','wb') as f:
    pickle.dump([vec_x,vec_y],f)

# # If sentence greater than 15 words
# for tok_sent in vec_x:
#     #tok_sent.insert(0,sentstart)
#     tok_sent.append(sentend)
#
# for tok_sent in vec_y:
#     #tok_sent.insert(0,sentstart)
#     tok_sent.append(sentend)
#
# max_num_words = 0
# for i,j in zip(vec_x,vec_y):
#     max_num_words=max(max_num_words,len(j),len(i))
# num_lines = len(vec_x)
# word_size = 300


#
# encoder_input_data = np.zeros(
#     (num_lines, max_num_words, word_size),
#     dtype='float32')
# decoder_input_data = np.zeros(
#     (num_lines, max_num_words, word_size),
#     dtype='float32')
# decoder_target_data = np.zeros(
#     (num_lines, max_num_words, word_size),
#     dtype='float32')
#
# for i in range(num_lines):
#     for j in range(len(vec_x[i])):
#         for k in range(word_size):
#             encoder_input_data[i][j][k] = vec_x[i][j][k]
#     for j in range(len(vec_y[i])):
#         for k in range(word_size):
#             decoder_input_data[i][j][k] = vec_y[i][j][k]
#             if j>0:
#                 decoder_target_data[i][j-1][k] = vec_y[i][j][k]
#
# embedding_matrix = np.zeros((len(model.vocab), word_size))
# for i in range(len(model.vocab)):
#     embedding_vector = model[model.index2word[i]]
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector
#
#
#
# vocab_size = len(model.index2word)
#
#
#
#
#
# encoder_inputs = Input(shape=(None,word_size))
# # x = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
# #                       weights=[embedding_matrix])(encoder_inputs)
# encoder_outputs,forward_h,forward_c,backward_h,backward_c = Bidirectional(LSTM(word_size,
#                            return_state=True))(encoder_inputs)
#
# state_h = Concatenate()([forward_h,backward_h])
# state_c = Concatenate()([forward_c,backward_c])
# encoder_states = [state_h, state_c]
#
# decoder_inputs = Input(shape=(None,word_size))
# # x = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
# #                       weights=[embedding_matrix])(decoder_inputs)
# decoder_lstm,_,_ = LSTM(word_size*2,return_sequences=True,return_state=True)(decoder_inputs,initial_state = encoder_states)
# decoder_outputs = Dense(word_size, activation='softmax')(decoder_lstm)
#
#
# # Define the model that will turn
# # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
# tmodel = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#
# # Compile & run training
# tmodel.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# # Note that `decoder_target_data` needs to be one-hot encoded,
# # rather than sequences of integers like `decoder_input_data`!
# tmodel.fit([encoder_input_data, decoder_input_data], decoder_target_data,
#           batch_size=64,
#           epochs=100,
#           validation_split=0.2)
#
# # Save model
# model.save('s2s.h5')
#
# # Next: inference mode (sampling).
# # Here's the drill:
# # 1) encode input and retrieve initial decoder state
# # 2) run one step of decoder with this initial state
# # and a "start of sequence" token as target.
# # Output will be the next target token
# # 3) Repeat with the current target token and current states
#
# # Define sampling models
# encoder_model = Model(encoder_inputs, encoder_states)
#
# decoder_state_input_h = Input(shape=(word_size,))
# decoder_state_input_c = Input(shape=(word_size,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_outputs, state_h, state_c = LSTM(word_size*2,return_sequences=True,return_state=True)(decoder_inputs,initial_state = encoder_states)
# decoder_states = [state_h, state_c]
# decoder_outputs = Dense(word_size, activation='softmax')(decoder_outputs)
# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs] + decoder_states)
#
# ########################################################################################################################
# def decode_sequence(input_seq):
#     # Encode the input as state vectors.
#     states_value = encoder_model.predict(input_seq)
#
#     # Generate empty target sequence of length 1.
#     target_seq = np.zeros((1, 1, word_size))
#     # Populate the first character of target sequence with the start character.
#     target_seq[0, 0, 0] = sentstart
#
#     # Sampling loop for a batch of sequences
#     # (to simplify, here we assume a batch of size 1).
#     stop_condition = False
#     decoded_sentence = ''
#     while not stop_condition:
#         output_tokens, h, c = decoder_model.predict(
#             [target_seq] + states_value)
#
#         # Sample a token
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         sampled_char = model.similar_by_vector[sampled_token_index]
#         decoded_sentence += sampled_char
#
#         # Exit condition: either hit max length
#         # or find stop character.
#         if (sampled_char == model.similar_by_vector(np.ones((300,),dtype=np.float32),topn=1)[0][0] or
#            len(decoded_sentence) > max_num_words):
#             stop_condition = True
#
#         # Update the target sequence (of length 1).
#         target_seq = np.zeros((1, 1, word_size))
#         target_seq[0, 0, sampled_token_index] = 1.
#
#         # Update states
#         states_value = [h, c]
#
#     return decoded_sentence
#
#
# while True:
#     # Take one sequence (part of the training set)
#     # for trying out decoding.
#
#     print('-')
#     ss = input('Input sentence:')
#     tok_ss = []
#     tok_ss.append(nltk.word_tokenize(ss.lower()))
#     sent_vec = [model[w] for w in tok_ss if w in model.vocab]
#     input_sentence = []
#     input_sentence.append(sent_vec)
#     input_sentence.insert(0, sentstart)
#     input_sentence.append(sentend)
#     decoded_sentence = decode_sequence(input_sentence)
#     print('Decoded sentence:', decoded_sentence)
