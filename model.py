
# coding: utf-8

# # Self-Driving Car ND : P3 - Behavioural Cloning 

# ## The below code building two different ConvNet to predict steering angle based on camera inputs. The first is a ConvNet based on LeNet architecture and the second is based on the NVIDIA model for autonomous driving. 

# ### Prepare the data

# In[1]:


import csv 
import cv2 
import numpy as np

path = 'data/IMG/'
images = [] 
measurements = [] 
with open('data/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    for line in reader: 

        correction = 0.25 #Tuning TBD
        steering = line[3]
        measurement = float(steering)
        
        # Center Image 
        filename = line[0].split('/')[-1]
        image = cv2.imread(path + filename)
        images.append(image)
        measurements.append(measurement)
        image_flipped = np.fliplr(image)
        measurement_flipped = -measurement
        images.append(image_flipped)
        measurements.append(measurement_flipped)

        # Left Image 
        filename = line[1].split('/')[-1]
        image = cv2.imread(path + filename)
        images.append(image)
        measurements.append(measurement + correction)
        image_flipped = np.fliplr(image)
        measurement_flipped = -(measurement + correction) 
        images.append(image_flipped)
        measurements.append(measurement_flipped)        
        
        # Right Image 
        filename = line[2].split('/')[-1]
        image = cv2.imread(path + filename)
        images.append(image)
        measurements.append(measurement - correction)
        image_flipped = np.fliplr(image)
        measurement_flipped = -(measurement - correction) 
        images.append(image_flipped)
        measurements.append(measurement_flipped)                
        
# # More data : Track 2 
# path = 'data2/IMG/'
# with open('data2/driving_log.csv') as csvfile: 
#     reader = csv.reader(csvfile)
#     for line in reader: 

#         correction = 0.25 #Tuning TBD
#         steering = line[3]
#         measurement = float(steering)
        
#         # Center Image 
#         filename = line[0].split('/')[-1]
#         image = cv2.imread(path + filename)
#         images.append(image)
#         measurements.append(measurement)
#         image_flipped = np.fliplr(image)
#         measurement_flipped = -measurement
#         images.append(image_flipped)
#         measurements.append(measurement_flipped)

#         # Left Image 
#         filename = line[1].split('/')[-1]
#         image = cv2.imread(path + filename)
#         images.append(image)
#         measurements.append(measurement + correction)
#         image_flipped = np.fliplr(image)
#         measurement_flipped = -(measurement + correction) 
#         images.append(image_flipped)
#         measurements.append(measurement_flipped)        
        
#         # Right Image 
#         filename = line[2].split('/')[-1]
#         image = cv2.imread(path + filename)
#         images.append(image)
#         measurements.append(measurement - correction)
#         image_flipped = np.fliplr(image)
#         measurement_flipped = -(measurement - correction) 
#         images.append(image_flipped)
#         measurements.append(measurement_flipped)                

        
# # More data : Track 2 
# path = 'data3/IMG/'
# with open('data3/driving_log.csv') as csvfile: 
#     reader = csv.reader(csvfile)
#     for line in reader: 

#         correction = 0.25 #Tuning TBD
#         steering = line[3]
#         measurement = float(steering)
        
#         # Center Image 
#         filename = line[0].split('/')[-1]
#         image = cv2.imread(path + filename)
#         images.append(image)
#         measurements.append(measurement)
#         image_flipped = np.fliplr(image)
#         measurement_flipped = -measurement
#         images.append(image_flipped)
#         measurements.append(measurement_flipped)

#         # Left Image 
#         filename = line[1].split('/')[-1]
#         image = cv2.imread(path + filename)
#         images.append(image)
#         measurements.append(measurement + correction)
#         image_flipped = np.fliplr(image)
#         measurement_flipped = -(measurement + correction) 
#         images.append(image_flipped)
#         measurements.append(measurement_flipped)        
        
#         # Right Image 
#         filename = line[2].split('/')[-1]
#         image = cv2.imread(path + filename)
#         images.append(image)
#         measurements.append(measurement - correction)
#         image_flipped = np.fliplr(image)
#         measurement_flipped = -(measurement - correction) 
#         images.append(image_flipped)
#         measurements.append(measurement_flipped)                

X = np.array(images)
y = np.array(measurements)


# In[2]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D

img_rows, img_cols, img_chls = X[1].shape


# ### LeNet Architecture

# In[ ]:


keep_prob = 0.2
batch_size = 128 
epochs = 5

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=X.shape[1:]))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(keep_prob))
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(keep_prob))
model.add(Dense(80))
model.add(Dropout(keep_prob))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam'
             )

model.fit(X, y,
          batch_size=batch_size,
          validation_split = 0.2,
          epochs=epochs,
          verbose=1, 
          shuffle=True
         )
          
model.save('model_lenet.h5')


# ### NVIDIA Architecture

# In[4]:


keep_prob = 0.4
batch_size = 128 
epochs = 4

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=X.shape[1:]))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(keep_prob))
model.add(Dense(32))
model.add(Dropout(keep_prob))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse',
              optimizer='adam'
             )

history_object = model.fit(X, y,
          batch_size=batch_size,
          validation_split = 0.2,
          epochs=epochs,
          verbose=1, 
          shuffle=True
         )

model.save('model_nvidia.h5')


# In[ ]:


from keras.models import Model
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()






