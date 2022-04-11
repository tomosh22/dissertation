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


if __name__ == "__main__":
	data = []
	with open("Electronics_5.json", "r") as file:
		contents = file.read()
		print("total number of reviews", len(contents.split("\n")))
		count = 0
		for x in contents.split("\n")[:50000]:
			count += 1
			if count == 10:
				count = 0
				continue
			data.append(x)
	jsonData = []
	for x in data[:-1]:
		jsonData.append(json.loads(x))
	random.shuffle(jsonData)
	reviews, labels = [], []
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
	for x in jsonData:
		review = x["reviewText"].lower()
		reviews.append(review)
		label = int(x["overall"])
		# if label >= 4:
		# 	labels.append(2)
		# elif label == 3:
		# 	labels.append(1)
		# else:
		# 	labels.append(0)
		labels.append(1 if label >= 4 else 0)
	reviews = np.array(reviews)
	labels = np.array(labels)
	num_classes = 2
	x_train, x_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.30)
	# y_train = keras.utils.to_categorical(y_train)
	# y_test = keras.utils.to_categorical(y_test)

	batch_size = 2

	# train dataset
	ds_train = zip(x_train, y_train)
	ds_test = zip(x_test, y_test)
	ds_train_encoded = encode_examples(ds_train).shuffle(len(x_train)).batch(batch_size)
	ds_test_encoded = encode_examples(ds_test).batch(batch_size)

	bertModel = TFBertModel.from_pretrained("bert-base-uncased")
	bertModel.summary()
	bertModel.layers[0].trainable = False
	input_ids = tf.keras.Input(shape=(120), dtype='int32')
	attention_masks = tf.keras.Input(shape=(120), dtype='int32')
	#token_type_ids_list = tf.keras.Input(shape=(70), dtype='int32')


	output = bertModel([input_ids, attention_masks])[0]
	output = Flatten()(output)
	for x in range(1):
		output = Dense(1, activation="relu")(output)
	output = Dropout(0.2)(output)
	output = Dense(2, activation="softmax")(output)
	model = models.Model(inputs=[input_ids, attention_masks], outputs=output)

	# output = Concatenate()([input_ids,attention_masks])
	# for x in range(5):
	# 	output = Dense(400, activation="relu")(output)
	# output = Dropout(0.2)(output)
	# output = Dense(2, activation="softmax")(output)
	# model = models.Model(inputs=[input_ids, attention_masks], outputs=output)

	model.summary()
	# recommended learning rate for Adam 5e-5, 3e-5, 2e-5
	learning_rate = 2e-5
	# multiple epochs might be better as long as we will not overfit the model
	number_of_epochs = 4

	# choosing Adam optimizer
	optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
	# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
	loss = tf.keras.losses.SparseCategoricalCrossentropy()
	metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

	model.compile(loss=loss,
				  optimizer=optimizer,
				  metrics=metric)

	history = model.fit(ds_train_encoded, batch_size=batch_size, epochs=10, validation_data=ds_test_encoded)