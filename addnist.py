import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Conv2D,MaxPooling2D
from sklearn import metrics
from sklearn.model_selection import train_test_split
#import tensorflow_addons as tfa
#from tensorflow_addons import layers as tfaLayers
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import pandas as pd


if __name__ == "__main__":
	input_shape = (28, 28, 3)

	x_train = np.load("train_x.npy")
	x_train = np.moveaxis(x_train, 1, -1)

	y_train = np.load("train_y.npy")
	x_valid = np.load("valid_x.npy")
	x_valid = np.moveaxis(x_valid, 1, -1)
	y_valid = np.load("valid_y.npy")
	x_test = np.load("test_x.npy")
	x_test = np.moveaxis(x_test, 1, -1)
	y_test = np.load("test_y.npy")
	y_train = keras.utils.to_categorical(y_train, 20)
	y_valid = keras.utils.to_categorical(y_valid, 20)
	y_test = keras.utils.to_categorical(y_test, 20)
	modelDeep = Sequential(
		[
			Input(shape=input_shape),
			Conv2D(32,3,padding="valid"),
			MaxPooling2D(),
			Conv2D(32,3,padding="valid"),
			MaxPooling2D(),
			Conv2D(64, 3, padding="valid"),
			MaxPooling2D(),
			Dropout(0.2),


			Dense(400,activation="relu"),
			Dense(400,activation="relu"),
			Dense(400,activation="relu"),
			Dense(400,activation="relu"),
			Dense(400,activation="relu"),
			Dense(400,activation="relu"),
			Dropout(0.2),
			Flatten(),
			Dense(20, activation="softmax"),
		]
	)

	modelShallow = keras.Sequential(
		[
			Input(shape=input_shape),
			Conv2D(32, 3, padding="valid"),
			MaxPooling2D(),
			Conv2D(32, 3, padding="valid"),
			MaxPooling2D(),
			Conv2D(64, 3, padding="valid"),
			MaxPooling2D(),
			Dropout(0.2),

			Dense(1000, activation="relu"),
			Dense(1000, activation="relu"),
			Dense(1000, activation="relu"),
			Dense(1000, activation="relu"),
			Dropout(0.2),
			Flatten(),
			Dense(20, activation="softmax"),
		]
	)

	modelDeep.summary()
	modelDeep.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=["accuracy"])
	modelDeep.fit(x_train, y_train, epochs=10, verbose=1, validation_split=0.15)
	scoreDeep = modelDeep.evaluate(x_test, y_test)
	print("Deep accuracy:", round(float(scoreDeep[1]), 3))

	modelShallow.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=["accuracy"])
	modelShallow.fit(x_train,y_train, epochs=10,verbose=1, validation_split=0.15)
	scoreShallow = modelShallow.evaluate(x_test, y_test)
	print("Shallow accuracy:", round(float(scoreShallow[1]),3))