from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import optimizers
from keras import backend as K
import tensorflow as tf
import numpy as np

img_width, img_height = 150, 150
top_model_weights_path = 'top_model_weights.h5'
train_data_dir = '../data/simulated_flight_1/train/'
valid_data_dir = '../data/simulated_flight_1/valid/'
nb_train_samples = 477
nb_valid_samples = 24
epochs = 20
batch_size = 3

def save_features():
  datagen = ImageDataGenerator(rescale = 1. / 255)

  vgg16_model = applications.VGG16(include_top = False, weights = 'imagenet')

  generator = datagen.flow_from_directory(train_data_dir, target_size = (img_width, img_height), batch_size = batch_size, class_mode = None, shuffle = False)
  top_model_features_train = vgg16_model.predict_generator(generator, nb_train_samples // batch_size)
  np.save(open('top_model_features_train.npy', 'wb'), top_model_features_train)

  generator = datagen.flow_from_directory(valid_data_dir, target_size = (img_width, img_height), batch_size = batch_size, class_mode = None, shuffle = False)
  top_model_features_valid = vgg16_model.predict_generator(generator, nb_valid_samples // batch_size)
  np.save(open('top_model_features_valid.npy', 'wb'), top_model_features_valid)

def triplet_loss(y_true, y_pred):
  alpha = 1
  embeddings = K.reshape(y_pred, (-1, 3))
  positive_distance = K.sqrt(K.sum((embeddings[:, 0] - embeddings[:, 1]) ** 2, axis = -1))
  negative_distance = K.sqrt(K.sum((embeddings[:, 0] - embeddings[:, 2]) ** 2, axis = -1))
  return K.sum(positive_distance - negative_distance + alpha)

def train_top_model():
  train_data = np.load(open('top_model_features_train.npy', 'rb'))
  train_labels = np.array([0] * (nb_train_samples))

  valid_data = np.load(open('top_model_features_valid.npy', 'rb'))
  valid_labels = np.array([0] * (nb_valid_samples))

  top_model = Sequential()
  top_model.add(Flatten(input_shape = train_data.shape[1:]))
  top_model.add(Dense(256, activation = 'relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(1, activation = 'sigmoid'))

  sgd_optimizer = optimizers.SGD(lr = 0.00001, decay = 1e-6, momentum = 0.9, nesterov = True)
  top_model.compile(optimizer = sgd_optimizer, loss = triplet_loss, metrics = ['accuracy'])
  top_model.fit(train_data, train_labels, epochs = epochs, batch_size = batch_size)
  top_model.save_weights(top_model_weights_path)

save_features()
train_top_model()

