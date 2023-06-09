import numpy as np
from AI_models import define_cnn
import os
import random
random.seed(42) 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from AI_utils import plot_loss, loss_split, signal_processing_2D
'load data'
path = 'DIS_lab_LoS/samples/'

filenames = os.listdir(path)
random.shuffle(filenames)

data = []
for f in filenames[:15000]: # because its too big, consider only 30,000 samples
    d = np.load(path+f)
    
    #splitting into complex parts
    data.append([d.real.reshape(64,100,1), d.imag.reshape(64,100,1)])
data = np.asarray(data)

x_train, x_test = train_test_split(data, test_size=0.05, random_state=42)

train_loss = []
val_loss = []
# average over multiple iterations
for i in range(5):
    print(f'Iteration {i}')
    'Lets define our model'
    act_f = 'tanh'
    model = define_cnn(act_f, lr = 1e-3)
    
    # Check if GPU is available
    gpu_available = tf.config.list_physical_devices('GPU')
    
    if gpu_available:
        print("GPU is available")
    else:
        print("GPU is not available")
    
    'Start training'
    with tf.device('/GPU:0'):
        tracker = model.fit(x_train, x_train, epochs = 30, batch_size = 32, validation_split = 0.05)

    train_loss.append(tracker.history['loss'])
    val_loss.append(tracker.history['val_loss'])


tr_avg, tr_upper, tr_lower = loss_split(train_loss)

val_avg, val_upper, val_lower = loss_split(val_loss)
    
plot_loss(tr_avg, tr_upper, tr_lower, val_avg, val_upper, val_lower)

'test the model'
x_pred = model.predict(x_test)
x_pred_rl = x_pred[:, 0, :, :, :].reshape(-1, 6400)
x_pred_im = x_pred[:, 1, :, :, :].reshape(-1, 6400)
x_true_rl = x_test[:, 0, :, :, :].reshape(-1, 6400)
x_true_im = x_test[:, 1, :, :, :].reshape(-1, 6400)

rl_mse = mean_squared_error(x_true_rl, x_pred_rl)
im_mse = mean_squared_error(x_true_im, x_pred_im)
print(f'real MSE {rl_mse}, imaginary MSE {im_mse}, total MSE {(rl_mse+im_mse) / 2}')

signal_processing_2D(x_true_rl + 1j*x_true_im, sample = 0, antenna = 0)
signal_processing_2D(x_pred_rl + 1j*x_pred_im, sample = 0, antenna = 0)

np.save('cnn_perf_ai_loss.npy', np.array(train_loss))
np.save('cnn_perf_ai_val_loss.npy', np.array(val_loss))
