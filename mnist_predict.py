from keras.preprocessing import image
import glob
import keras
# import os
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Use the model wa train by mnist_example.py to predict the picture you want
# set file path of the data you want to predict
file_path = 'DATA/'
f_names = glob.glob(file_path + '*.jpg')
img = []

# "mnist_model.h5" has the weight and parameter that you have already trained
#  or maybe you can just think "mnist_model.h5" is the model you can taken to predict

# (Keras function) Load model by the "mnist_model.h5"
model = keras.models.load_model('mnist_model.h5')
for i in range(len(f_names)):
    # load every figures in the "file_path" you set
    images = image.load_img(f_names[i], target_size=(28,28),color_mode = "grayscale")
    # translate figure to array that we need for calculating during predicting
    x = image.img_to_array(images)
    # let x be the same size we need for model
    x = np.expand_dims(x, 0)
    print('loading no.%s image' % i)

    # print every figure's label we predicted by "mnist_model.h5"
    y = model.predict(x)
    print(y)

