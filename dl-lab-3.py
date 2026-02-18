import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test)=tf.keras.datasets.cifar10.load_data()

print(x_train.shape) 
print(y_train.shape)

x_train=x_train/255.0
x_test=x_test/255.0

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))
model.summary()

history=model.fit(
    x_train,y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test)
)


test_loss,test_acc=model.evaluate(x_test, y_test)
print("Test Accuracy:",test_acc)

pred=model.predict(x_test)

plt.imshow(x_test[0])
plt.show()

print("Predicted:",np.argmax(pred[0]))
print("Actual:",y_test[0])
