import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding, GlobalAveragePooling1D,Dropout,Conv2D,MaxPooling2D, Concatenate
from sklearn import metrics
from sklearn.model_selection import train_test_split
# import tensorflow_addons as tfa
# from tensorflow_addons import layers as tfaLayers
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertForSequenceClassification, TFBertModel
from transformers import InputExample, InputFeatures
import random
from tensorflow.keras import models
from keras.datasets import imdb
from keras.preprocessing import sequence


# map to the expected input to TFBertForSequenceClassification
def map_example_to_dict(input_ids, attention_masks, label):
	return {
			   "input_1": input_ids,
			   #"input_2": token_type_ids,
			   "input_2": attention_masks,
		   }, label


def encode_examples(ds):
	# prepare list, so that we can build up final TensorFlow dataset from slices.
	input_ids_list = []
	token_type_ids_list = []
	attention_mask_list = []
	label_list = []
	for reviewInLoop, label in ds:
		bert_input = tokenizer.encode_plus(reviewInLoop,
										   add_special_tokens=True,  # add [CLS], [SEP]
										   max_length=120,  # max length of the text that can go to BERT
										   padding='max_length',
										   truncation=True,
										   return_attention_mask=True,  # add attention mask to not focus on pad tokens
										   )

		input_ids_list.append(bert_input['input_ids'])
		token_type_ids_list.append(bert_input['token_type_ids'])
		attention_mask_list.append(bert_input['attention_mask'])
		label_list.append([label])

	return tf.data.Dataset.from_tensor_slices(
		(input_ids_list, attention_mask_list, label_list)).map(map_example_to_dict)


if __name__ =="__main__":
	(x_train, y_train), (x_test,y_test) = imdb.load_data(num_words=20000)
	x_train = sequence.pad_sequences(x_train, maxlen=100)
	x_test = sequence.pad_sequences(x_test, maxlen=100)
	print(x_train.shape)
	print(y_train.shape)
	model = Sequential([
		Embedding(20000, 20, input_length=100),
		GlobalAveragePooling1D(),
		Dense(110, activation="relu"),
		Dropout(0.2),
		Dense(1,activation="sigmoid")
	])
	model.summary()
	model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
	model.fit(x_train,y_train, epochs=15, verbose=1,validation_data=(x_test, y_test))

if False:
	max_features = 20000
	maxlen = 80  # cut texts after this number of words (among top max_features most common words)
	batch_size = 32

	print('Loading data...')
	(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
	print(len(x_train), 'train sequences')
	print(len(x_test), 'test sequences')

	print('Pad sequences (samples x time)')
	x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
	x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
	print('x_train shape:', x_train.shape)
	print('x_test shape:', x_test.shape)

	print('Build model...')
	model = Sequential()
	model.add(Embedding(max_features, 128))
	model.add(GlobalAveragePooling1D())
	model.add(Dense(500))
	model.add(Dense(1, activation='sigmoid'))

	# try using different optimizers and different optimizer configs
	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	print('Train...')
	model.fit(x_train, y_train,
			  batch_size=batch_size,
			  epochs=15,
			  validation_data=(x_test, y_test))
	score, acc = model.evaluate(x_test, y_test,
								batch_size=batch_size)
	print('Test score:', score)
	print('Test accuracy:', acc)