from keras import applications, optimizers
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Input, Dropout, Flatten, Dense
import numpy as np

def triplet_loss(y_true, y_pred):
  alpha = 1
  embeddings = K.reshape(y_pred, (-1, 3))
  positive_distance = K.sqrt(K.sum((embeddings[:, 0] - embeddings[:, 1]) ** 2, axis = -1))
  negative_distance = K.sqrt(K.sum((embeddings[:, 0] - embeddings[:, 2]) ** 2, axis = -1))
  return K.sum(positive_distance - negative_distance + alpha)

img_width, img_height = 150, 150
top_model_weights_path = 'top_model_weights.h5'
train_data_dir = '../data/simulated_flight_1/train/'
valid_data_dir = '../data/simulated_flight_1/valid/'
nb_train_samples = 477
nb_valid_samples = 24
epochs = 24
batch_size = 15
input_tensor = Input(shape = (150, 150, 3))

vgg16_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

train_data = np.load(open('top_model_features_train.npy', 'rb'))

top_model = Sequential()
top_model.add(Flatten(input_shape = vgg16_model.output_shape[1:]))
top_model.add(Dense(256, activation = 'relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation = 'sigmoid'))

top_model.load_weights(top_model_weights_path)

model = Sequential()
for l in vgg16_model.layers:
  model.add(l)
model.add(top_model)

# set some layers to be non trainable
for layer in model.layers[:25]:
  layer.trainable = False

model.compile(loss = triplet_loss, optimizer = optimizers.SGD(lr = 1e-4, momentum = 0.9), metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1. / 255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
valid_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size = (img_height, img_width), batch_size = batch_size, class_mode = 'binary')
valid_generator = valid_datagen.flow_from_directory(valid_data_dir, target_size = (img_height, img_width), batch_size = batch_size, class_mode = 'binary')

model.fit_generator(train_generator, samples_per_epoch = nb_train_samples, epochs = epochs, validation_data = valid_generator, nb_val_samples = nb_valid_samples)

