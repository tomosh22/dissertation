import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,MaxPooling2D,Conv2D,GlobalAveragePooling2D
from sklearn import metrics
from sklearn.model_selection import train_test_split
#import tensorflow_addons as tfa
#from tensorflow_addons import layers as tfaLayers
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import pandas as pd
from keras import models


if __name__ == "__main__":
	(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
	x_train = x_train.astype("float32") / 255
	x_test = x_test.astype("float32") / 255
	y_train = keras.utils.to_categorical(y_train)
	print(y_train)
	y_test = keras.utils.to_categorical(y_test)
	input_shape=(32,32,3)
	modelDeep = Sequential(
		[
			Input(shape=input_shape),
			Conv2D(32, 3, padding="valid"),
			MaxPooling2D(),
			Conv2D(32, 3, padding="valid"),
			MaxPooling2D(),
			Conv2D(64, 3, padding="valid"),
			MaxPooling2D(),
			Dropout(0.2),

			Dense(1500,activation="relu"),
			Dense(1500,activation="relu"),
			Dense(1500,activation="relu"),
			Dense(1500,activation="relu"),
			Dropout(0.2),
			Flatten(),
			Dense(10, activation="softmax"),
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
			Dropout(0.2),
			Flatten(),
			Dense(10, activation="softmax"),
		]
	)

	baseModel = tf.keras.applications.ResNet50(weights="imagenet",input_shape=input_shape,include_top=False)
	x = baseModel.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(3000, activation='relu')(x)
	x = Dense(3000, activation='relu')(x)
	x = Dense(10,activation="softmax")(x)
	model = models.Model(inputs=baseModel.input, outputs=x)
	model.summary()
	model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=["accuracy"])
	model.fit(x_train, y_train, epochs=10, verbose=1, validation_split=0.15)


	modelDeep.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=["accuracy"])
	modelShallow.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=["accuracy"])
	modelDeep.summary()

	modelDeep.fit(x_train,y_train, epochs=10,verbose=1, validation_split=0.15)
	modelShallow.fit(x_train,y_train, epochs=10,verbose=1, validation_split=0.15)

	scoreDeep = modelDeep.evaluate(x_test, y_test)
	scoreShallow = modelShallow.evaluate(x_test, y_test)
	print("Deep accuracy:", round(float(scoreDeep[1]),3))
	print("Shallow accuracy:", round(float(scoreShallow[1]),3))