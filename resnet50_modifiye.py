
import numpy as np
import os
import time
from resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from imagenet_utils import preprocess_input, decode_predictions
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split



PATH = os.getcwd()
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('veriler yüklendi->>'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img 
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)



num_classes = 4
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:202]=0
labels[202:404]=1
labels[404:606]=2
labels[606:]=3

names = ['kedi','kopek','at','insan']

Y = np_utils.to_categorical(labels, num_classes)


x,y = shuffle(img_data,Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

image_input = Input(shape=(224, 224, 3))
#imagenet üzerinde eğitilmiş resnet50 ağını kullanacağız
#ilk param. resimin matris tanımı ve ağırlık olarak imagenet üzerindeki ağırlığı kullanacağımızı belirtiyoruz
#bu aynı zaman tecrübe aktarımı olarak tanımlanabilir
model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()
#önceden tanımlı resnet50 ağını kendimize göre modifiye ediyoruz
last_layer = model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)
custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()

#yeni ağı yeniden eğitebilmek için tüm katmanların eğitim moduna getiriyoruz
for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

custom_resnet_model.layers[-1].trainable

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#eğitim
t=time.time()
hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=1, verbose=1, validation_data=(X_test, y_test))
print('eğitim süresi: %s' % (time.time()-t))
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#### test

img_path = 'kedi.jpg'
img = image.load_img(img_path, target_size=(224, 224))

test_img = image.img_to_array(img)
test_img = np.expand_dims(test_img, axis=0)
test_img = preprocess_input(test_img)

preds = model.predict(test_img)
print('Tahmin:', decode_predictions(preds))


