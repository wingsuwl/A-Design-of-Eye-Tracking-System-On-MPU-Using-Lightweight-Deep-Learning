import numpy as np
import matplotlib
import h5py
import os, shutil
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras import layers, models, optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import SqueezeNet

import tensorflow as tf

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense


###############################################################################
# Diretory Path Creation
# Can separate this code to different python file and just import to this
#original_dataset_dir = '/Users/jisuk/OneDrive/바탕 화면/datasets/catsAndDogs/train'
base_dir = '/home/jisukim/eye1001/datasets/eyesmall6'

if os.path.exists(base_dir):  # 반복적인 실행을 위해 디렉토리를 삭제합니다.
    shutil.rmtree(base_dir)   # 이 코드는 책에 포함되어 있지 않습니다.
os.mkdir(base_dir)
# 훈련, 검증, 테스트 분할을 위한 디렉터리
train_dir = os.path.join(base_dir, 'train')
# os.path.join = base_dir에 선언된 주소에  'train' 이라는 폴더를 생성 
# ./datasets/cats_and_dogs_small/train 

os.mkdir(train_dir)
# train_dir이라는 경로를 실제로 make directory함 

validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# 총 트레인, 검증, 테스트라는 폴더를 생성

# 훈련용
train_forward_dir = os.path.join(train_dir, 'forward')
os.mkdir(train_forward_dir)

train_closed_dir = os.path.join(train_dir, 'side')
os.mkdir(train_closed_dir)


# 검증용
validation_forward_dir = os.path.join(validation_dir, 'forward')
os.mkdir(validation_forward_dir)

validation_closed_dir = os.path.join(validation_dir, 'side')
os.mkdir(validation_closed_dir)


# 테스트용
test_forward_dir = os.path.join(test_dir, 'forward')
os.mkdir(test_forward_dir)

test_closed_dir = os.path.join(test_dir, 'side')
os.mkdir(test_closed_dir)

#################################side#####################################	
	
fnames = ['{}.JPG'.format(i) for i in range(501,1001)]
for fname in fnames:
    src = os.path.join("/home/jisukim/eye1001/datasets/eye6/eye/train/side", fname)
    dst = os.path.join(train_closed_dir, fname)
    shutil.copyfile(src, dst)	
	
	
fnames = ['{}.JPG'.format(i) for i in range(201,501)]
for fname in fnames:
    src = os.path.join("/home/jisukim/eye1001/datasets/eye6/eye/validation/side", fname)
    dst = os.path.join(validation_closed_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.JPG'.format(i) for i in range(1,201)]
for fname in fnames:
    src = os.path.join("/home/jisukim/eye1001/datasets/eye6/eye/test/side", fname)
    dst = os.path.join(test_closed_dir, fname)
    shutil.copyfile(src, dst)

###############################forward#######################################

fnames = ['{}.JPG'.format(i) for i in range(501,1001)]
for fname in fnames:
    src = os.path.join("/home/jisukim/eye1001/datasets/eye6/eye/train/forward", fname)
    dst = os.path.join(train_forward_dir, fname)
    shutil.copyfile(src, dst)
	
fnames = ['{}.JPG'.format(i) for i in range(201,501)]
for fname in fnames:
    src = os.path.join("/home/jisukim/eye1001/datasets/eye6/eye/validation/forward", fname)
    dst = os.path.join(validation_forward_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['{}.JPG'.format(i) for i in range(1,201)]
for fname in fnames:
    src = os.path.join("/home/jisukim/eye1001/datasets/eye6/eye/test/forward", fname)
    dst = os.path.join(test_forward_dir, fname)
    shutil.copyfile(src, dst)
	
########################################################################################
# Start learning, as well as compiling model
sn = SqueezeNet(input_shape = (100, 75, 3), nb_classes=2)
'''
sn = Sequential()
sn.add(Flatten(input_shape=train_data.shape[1:]))
sn.add(Dense(256, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)))
sn.add(Dropout(0.3))
sn.add(BatchNormalization())
sn.add(Dense(2, activation='softmax'))
'''
sn.summary()
train_data_dir = '/home/jisukim/eye1001/datasets/eyesmall6/train'
validation_data_dir = '/home/jisukim/eye1001/datasets/eyesmall6/validation'
test_data_dir  = '/home/jisukim/eye1001/datasets/eyesmall6/test'
train_samples = 1000
validation_samples = 600
epochs = 50
nb_class = 2
width, height = 100, 75

sgd = SGD(lr=0.001, decay=0.0002, momentum=0.9, nesterov=True)
sn.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.FalsePositives(name='false_positives'),tf.keras.metrics.FalseNegatives(name='false_negatives')])

#   Generator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(width, height),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(width, height),
        batch_size=32,
        class_mode='categorical')
		
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(width, height),  
        batch_size=32,
        class_mode='categorical')
############################################################################
# Inlcude this Callback checkpoint if you want to make .h5 checkpoint files
# May slow your training
#early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
#checkpoint = ModelCheckpoint(                                         
#                'weights.{epoch:02d}-{val_loss:.2f}.h5',
#                monitor='val_loss',                               
#                verbose=0,                                        
#                save_best_only=True,                              
#                save_weights_only=True,                           
#                mode='min',                                       
#                period=1)                                
###########################################################################

hist=sn.fit_generator(
        train_generator,
        steps_per_epoch=train_samples,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples 
        #,callbacks=[checkpoint]
)

#########################################################################################33

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'g', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'r', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'b', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='lower left')
acc_ax.legend(loc='upper left')

plt.savefig('ourmodel2.png')

###########################################################################

print("-- Evaluate --")
scores = sn.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(sn.metrics_names[1], scores[1]*100))

print("-- Predict --")
output = sn.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

print(test_generator.class_indices)
print(output)

print("Training Ended")

sn.save_weights('ourweights2.h5')
print("Saved weight file")

sn.save('ourmodel2.h5')
print("saved model file")

# End of Code
##########################################################################