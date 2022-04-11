import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Conv2D,MaxPooling2D,GlobalAveragePooling2D,RandomFlip, RandomRotation
from sklearn import metrics
from sklearn.model_selection import train_test_split
#import tensorflow_addons as tfa
#from tensorflow_addons import layers as tfaLayers
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import pandas as pd
from tensorflow.keras import models
import matplotlib.pyplot as plt


if __name__ == "__main__":
	input_shape = (198,253, 3)
	dataset = tf.keras.utils.image_dataset_from_directory("goodguysbadguys/train", subset="training", validation_split=0.2,image_size=(198,253),seed=1000)
	dataset = dataset.shuffle(1000)
	print(dataset)
	for images, labels in dataset.take(1):

		plt.imshow(images[0].numpy().astype("uint8"))
		plt.axis("off")
	plt.show()
	print(dataset)
	#baseModel = tf.keras.applications.InceptionV3(weights="imagenet",input_shape=input_shape,include_top=False)
	#x = baseModel.output
	#x = GlobalAveragePooling2D()(x)
	#x = Dense(2,activation="softmax")(x)
	#model = models.Model(inputs=baseModel.input, outputs=x)
	model = Sequential(
		[
			Input(shape=input_shape),
			RandomFlip("horizontal_and_vertical"),
			RandomRotation(0.3),
			Conv2D(8, 3, padding="valid"),
			MaxPooling2D(),
			Conv2D(8, 3, padding="valid"),
			MaxPooling2D(),
			Conv2D(8, 3, padding="valid"),
			MaxPooling2D(),
			Conv2D(16, 3, padding="valid"),
			MaxPooling2D(),
			Conv2D(16, 3, padding="valid"),
			MaxPooling2D(),
			Conv2D(16, 3, padding="valid"),
			MaxPooling2D(),
			Dropout(0.2),
			Flatten(),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dense(200, activation="relu"),
			Dropout(0.2),
			Flatten(),
			Dense(2, activation="softmax"),
		]#79.4
	)
	model.summary()
	model.compile(loss=keras.losses.sparse_categorical_crossentropy,
				  optimizer=keras.optimizers.Adam(),
				  metrics=['accuracy'])
	history = model.fit(dataset,epochs=20, verbose=1)
	plt.plot(history.history["accuracy"])
	plt.title("25x200")
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.show()

	#1x17 0.7977
	#20x50 0.8288
	#10x200 0.8254
	#5x1000 0.8156
	#2x100 0.8211
	#5x100 0.8108
	#25x200 stuck at 0.5