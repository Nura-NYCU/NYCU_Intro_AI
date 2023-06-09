import numpy as np
from VAE_CNN_MODEL import define_vae
import os
import random
random.seed(42) 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from AI_utils import plot_loss, loss_split, signal_processing_2D


record = True
save_models = False

'load data'
path = 'DIS_lab_LoS/samples/'

filenames = os.listdir(path)
random.shuffle(filenames)

data = []
for f in filenames[:10000]: # because its too big, consider only 30,000 samples
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
    vae_encoder_imag, vae_decoder_imag, vae_encoder_real, vae_decoder_real, vae_imag, vae_real = define_vae()
    
    # Check if GPU is available
    gpu_available = tf.config.list_physical_devices('GPU')
    
    if gpu_available:
        print("GPU is available")
    else:
        print("GPU is not available")
        
    'Start training'
    tr_loss = []
    vl_loss = []
    for ep in range(71):
        with tf.device('/GPU:0'):
            tracker = vae_real.fit(x_train[:, 0, :, :, :], epochs = 1, batch_size = 16)
            tracker = vae_imag.fit(x_train[:, 1, :, :, :], epochs = 1, batch_size = 16)
        # train_loss.append(tracker.history['loss'])
        if ep % 10 == 0 and save_models:
            vae_encoder_imag.save('AI_models/vae_enc_img') 
            vae_decoder_imag.save('AI_models/vae_dec_img') 
            
            vae_encoder_real.save('AI_models/vae_enc_rl') 
            vae_decoder_real.save('AI_models/vae_dec_rl')
            
        if ep % 10 == 0 and record:
            _,_,z = vae_encoder_real.predict(x_train[:,0,:,:,:])
            pred_rl = vae_decoder_real.predict(z)
            
            _,_,z = vae_encoder_imag.predict(x_train[:,1,:,:,:])
            pred_im = vae_decoder_imag.predict(z)
            
            true_rl = x_train[:, 0, :, :, :].reshape(-1, 6400)
            pred_rl = pred_rl.reshape(-1, 6400)
            mse_rl = mean_squared_error(true_rl, pred_rl)
            
            true_im = x_train[:, 1, :, :, :].reshape(-1, 6400)
            pred_im = pred_im.reshape(-1, 6400)
            mse_im = mean_squared_error(true_im, pred_im)
            
            tr_mse = (mse_rl + mse_im) / 2
            
            del true_im
            del true_rl
            del pred_im
            del pred_rl
            del z
    
            _,_,z = vae_encoder_real.predict(x_test[:,0,:,:,:])
            pred_rl = vae_decoder_real.predict(z)
            
            _,_,z = vae_encoder_imag.predict(x_test[:,1,:,:,:])
            pred_im = vae_decoder_imag.predict(z)
            
            true_rl = x_test[:, 0, :, :, :].reshape(-1, 6400)
            pred_rl = pred_rl.reshape(-1, 6400)
            mse_rl = mean_squared_error(true_rl, pred_rl)
            
            true_im = x_test[:, 1, :, :, :].reshape(-1, 6400)
            pred_im = pred_im.reshape(-1, 6400)
            mse_im = mean_squared_error(true_im, pred_im)
            
            val_mse = (mse_rl + mse_im) / 2
            
            tr_loss.append(tr_mse)
            vl_loss.append(val_mse)
    if record:
        train_loss.append(tr_loss)
        val_loss.append(vl_loss)

if record:
    tr_avg, tr_upper, tr_lower = loss_split(train_loss)
    
    val_avg, val_upper, val_lower = loss_split(val_loss)
        
    plot_loss(tr_avg, tr_upper, tr_lower, val_avg, val_upper, val_lower , xtick = 1)

'test the model'
_,_,z = vae_encoder_real.predict(x_test[:,0,:,:,:])
pred_rl = vae_decoder_real.predict(z)

_,_,z = vae_encoder_imag.predict(x_test[:,1,:,:,:])
pred_im = vae_decoder_imag.predict(z)

true_rl = x_test[:, 0, :, :, :].reshape(-1, 6400)
pred_rl = pred_rl.reshape(-1, 6400)
mse_rl = mean_squared_error(true_rl, pred_rl)

true_im = x_test[:, 1, :, :, :].reshape(-1, 6400)
pred_im = pred_im.reshape(-1, 6400)
mse_im = mean_squared_error(true_im, pred_im)

val_mse = (mse_rl + mse_im) / 2
vl_loss.append(val_mse)

print(f'real MSE {mse_rl}, imaginary MSE {mse_im}, total MSE {val_mse}')

signal_processing_2D(true_rl + 1j*true_im, sample = 0, antenna = 0)
signal_processing_2D(pred_rl + 1j*pred_im, sample = 0, antenna = 0)

np.save('vae_perf_ai_loss.npy', np.array(train_loss))
# np.save('vae_perf_ai_val_loss.npy', np.array(val_loss))
