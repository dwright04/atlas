import numpy as np
import scipy.io as sio

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint  

from analysis import roc_curve

def one_percent_mdr(y_true, y_pred):
  t = 0.01
  fpr, tpr, thresholds = roc_curve(y_true, y_pred, step=0.001)
  return fpr[np.where(1-tpr<=t)[0]][-1]

def one_percent_fpr(y_true, y_pred):
  t = 0.01
  fpr, tpr, thresholds = roc_curve(y_true, y_pred, step=0.001)
  return 1-tpr[np.where(fpr<=t)[0]][0]

def load_data(filename):
  data = sio.loadmat(filename)

  X = data['X']
  y_train = np.squeeze(data['y'])
  train_files = np.squeeze(data['train_files'])
  m, n = X.shape
  image_dim = int(np.sqrt(n))
  x_train = np.zeros((m, image_dim, image_dim, 1))
  for i in range(m):
    x_train[i,:,:,0] += np.reshape(X[i], (image_dim, image_dim), order='F')

  X = data['testX']
  y_test = np.squeeze(data['testy'])
  test_files = np.squeeze(data['test_files'])
  m, n = X.shape
  x_test = np.zeros((m, image_dim, image_dim, 1))
  for i in range(m):
    x_test[i,:,:,0] += np.reshape(X[i], (image_dim, image_dim), order='F')
  
  return (x_train, y_train, train_files), (x_test, y_test, test_files), image_dim

def create_model(num_classes, image_dim):
  model = Sequential()
  model.add(Conv2D(filters=16, kernel_size=2, padding='same', \
                   activation='relu', input_shape=(image_dim, image_dim, 1)))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(0.3))
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.4))
  model.add(Dense(num_classes, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', \
                kerasmetrics=['accuracy'])
  print(model.summary())
  return model

def main():

  path = ''
  filename = 'andrei_20x20_skew3_signpreserve_f200000b600000.mat'
  train_data, test_data, image_dim = load_data(path + filename)

  num_classes = 2

  x_train = train_data[0]
  y_train = np_utils.to_categorical(train_data[1], num_classes)

  m = x_train.shape[0]
  split_frac = int(.75*m)
  (x_train, x_valid) = x_train[:split_frac], x_train[split_frac:]
  (y_train, y_valid) = y_train[:split_frac], y_train[split_frac:]

  x_test = test_data[0]
  #y_test = np_utils.to_categorical(test_data[1], num_classes)
  y_test = test_data[1]

  model = create_model(num_classes, image_dim)
  """  
  checkpointer = ModelCheckpoint(filepath='atlas.model.best.hdf5', \
                                 verbose=1, save_best_only=True)

  model.fit(x_train, y_train, batch_size=128, epochs=20, \
            validation_data=(x_valid, y_valid), \
            callbacks=[checkpointer], verbose=1, shuffle=True)
  """
  model.load_weights('atlas.model.best.hdf5')

  (y_train, y_valid) = train_data[1][:split_frac], train_data[1][split_frac:]

  print('[+] Training Set Error:')
  pred = model.predict(x_train, verbose=0)
  print(one_percent_mdr(y_train, pred[:,1]))
  print(one_percent_fpr(y_train, pred[:,1]))

  print('[+] Validation Set Error:')
  pred = model.predict(x_valid, verbose=0)
  print(one_percent_mdr(y_valid, pred[:,1]))
  print(one_percent_fpr(y_valid, pred[:,1]))

  print('[+] Test Set Error:')
  pred = model.predict(x_test, verbose=0)
  print(one_percent_mdr(y_test, pred[:,1]))
  print(one_percent_fpr(y_test, pred[:,1]))

  output = open("tmp.csv","w")
  for i in range(len(pred[:,1])):
    output.write("%s,%E,%d,\n"%(test_data[2][i], pred[i,1], y_test[i]))
  output.close()

if  __name__ == '__main__':
  main()
