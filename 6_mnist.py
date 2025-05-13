import tensorflow as tf 
import numpy as np 
import pandas as pd   
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

x_train=x_train.reshape(-1,28,28,1).astype('float32')
x_test=x_test.reshape(-1,28,28,1).astype('float32')
x_train/=255
x_test/=255
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=32,epochs=20,validation_data=[x_test,y_test])

y_pred=model.predict(x_test)

y_test_class=np.argmax(y_test,axis=1)
y_pred_class=np.argmax(y_pred,axis=1)

ac=accuracy_score(y_pred_class,y_test_class)
cr=classification_report(y_pred_class,y_test_class)
cm=confusion_matrix(y_pred_class,y_test_class)
f1=f1_score(y_pred_class,y_test_class,average='macro')
print(ac)
print(f1)
print(cr)
print(cm)
