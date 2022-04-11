import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout
from sklearn import metrics
from sklearn.model_selection import train_test_split
#import tensorflow_addons as tfa
#from tensorflow_addons import layers as tfaLayers
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import pandas as pd


if __name__ == "__main__":
	df = pd.read_csv("isolet_csv.csv")
	#1 row = one sound
	labels = df.pop("class")
	features = df
	features.columns = [c.replace("f", "") for c in list(features.columns)]
	features.columns = features.columns.astype(int)
	#dropIndexes = list(range(1,features.shape[1],1))
	#features = features.drop(columns=dropIndexes)
	#features = features[dropIndexes]
	print(features)
	print(features.shape)
	labels = np.array(labels).astype('U')
	labels = np.char.replace(labels,"'","")

	labels = labels.astype(int)
	print(labels)
	print("labels shape",labels.shape)


	features = np.array(features)

	print("features shape",features.shape)
	input_shape = (617)
	x_train, x_test, y_train, y_test = train_test_split(features,labels,test_size=0.1)
	print(y_train)
	print()
	y_train = keras.utils.to_categorical(y_train)
	print(y_train)
	y_test = keras.utils.to_categorical(y_test)

	modelDeep = Sequential(
		[
			Input(shape=input_shape),
			Dense(1000,activation="relu"),
			Dense(1000,activation="relu"),
			Dense(1000,activation="relu"),
			Dense(1000,activation="relu"),
			Dropout(0.2),
			Dense(27, activation="softmax"),
		]
	)

	modelShallow = keras.Sequential(
		[
			Input(shape=input_shape),
			Dense(1000, activation="relu"),
			Dropout(0.2),
			Dense(27, activation="softmax"),
		]
	)

	modelDeep.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=["accuracy"])
	modelShallow.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=["accuracy"])
	modelDeep.summary()

	modelDeep.fit(x_train,y_train, epochs=10,verbose=1, validation_split=0.15)
	modelShallow.fit(x_train,y_train, epochs=10,verbose=1, validation_split=0.15)

	scoreDeep = modelDeep.evaluate(x_test, y_test)
	scoreShallow = modelShallow.evaluate(x_test, y_test)
	print("Deep accuracy:", round(float(scoreDeep[1]),3))
	print("Shallow accuracy:", round(float(scoreShallow[1]),3))