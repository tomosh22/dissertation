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
	input_shape = (135,240, 3)
	dataset = tf.keras.utils.image_dataset_from_directory("cats", subset="training", validation_split=0.2,image_size=(135,240),seed=1000)
	dataset = dataset.shuffle(1000)
	print(dataset)
	for images, labels in dataset.take(1):
		for image in images:
			plt.imshow(image.numpy().astype("uint8"))
			plt.axis("off")
			plt.show()
	print(dataset)
	#baseModel = tf.keras.applications.InceptionV3(weights="imagenet",input_shape=input_shape,include_top=False)
	#x = baseModel.output
	#x = GlobalAveragePooling2D()(x)
	#x = Flatten(x)
	#x = Dense(1,activation="relu")(x)
	#x = Dense(2,activation="softmax")(x)
	#model = models.Model(inputs=baseModel.input, outputs=x)
	model = Sequential(
		[
			Input(shape=input_shape),
			RandomFlip("horizontal_and_vertical"),
			RandomRotation(0.3),
			Conv2D(16, 2, padding="valid"),
			MaxPooling2D(),
			Conv2D(16, 2, padding="valid"),
			MaxPooling2D(),
			Conv2D(16, 2, padding="valid"),
			MaxPooling2D(),
			Conv2D(16, 2, padding="valid"),
			MaxPooling2D(),
			Conv2D(16, 2, padding="valid"),
			MaxPooling2D(),
			Conv2D(16, 2, padding="valid"),
			MaxPooling2D(),
			Dropout(0.2),
			Flatten(),
			Dense(33, activation="relu"),
			Dropout(0.2),
			Flatten(),
			Dense(2, activation="softmax"),
		]
	)
	model.summary()
	model.compile(loss=keras.losses.sparse_categorical_crossentropy,
				  optimizer=keras.optimizers.Adam(),
				  metrics=['accuracy'])
	model.fit(dataset,epochs=5, verbose=1)


