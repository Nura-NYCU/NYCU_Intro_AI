import numpy as np
from AI_models import define_vae
import os
import random
random.seed(42) 
# from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_squared_error
# from AI_utils import plot_loss, loss_split, signal_processing_2D


record = False
save_models = True

'load data'
path = 'DIS_lab_LoS/samples/'

filenames = os.listdir(path)
random.shuffle(filenames)

# Artificial clustering
# Cluster 1 --> L2 norm in range 38-44
# Cluster 2 --> L2 norm in range 48-54
# Cluster 3 --> L2 norm in range 58-64
max_size = 10000
data = []
norms = [[], [], []]
for f in filenames: # because its too big, consider only 30,000 samples
    d = np.load(path+f)
    norm = np.linalg.norm(d)
    if norm >=38 and norm <44 and len(norms[0]) < max_size:
        norms[0].append(f)
    if norm >=48 and norm <54 and len(norms[1]) < max_size:
        norms[1].append(f)
    if norm >=58 and norm <64 and len(norms[2]) < max_size:
        norms[2].append(f)

for i in norms:
    print(f'Length of each cluster dataset {len(i)}')
    
data = [[],[],[]] #dataset of clusters 

for i, cluster in enumerate(norms):
    for f in cluster:
        d = np.load(path+f)
        data[i].append([d.real.reshape(64,100,1), d.imag.reshape(64,100,1)])
    data[i] = np.asarray(data[i])

print(f'Iteration {i}')
'Lets define our model'

vae_encoder_imag1, vae_decoder_imag1, vae_encoder_real1, vae_decoder_real1, vae_imag1, vae_real1 = define_vae()
vae_encoder_imag2, vae_decoder_imag2, vae_encoder_real2, vae_decoder_real2, vae_imag2, vae_real2 = define_vae()
vae_encoder_imag3, vae_decoder_imag3, vae_encoder_real3, vae_decoder_real3, vae_imag3, vae_real3 = define_vae()

# Check if GPU is available
gpu_available = tf.config.list_physical_devices('GPU')

if gpu_available:
    print("GPU is available")
else:
    print("GPU is not available")
    
'Start training'
for ep in range(71):
    with tf.device('/GPU:0'):
        vae_real1.fit(data[0][:, 0, :, :, :], epochs = 1, batch_size = 16)
        vae_imag1.fit(data[0][:, 1, :, :, :], epochs = 1, batch_size = 16)
        
        vae_real2.fit(data[1][:, 0, :, :, :], epochs = 1, batch_size = 16)
        vae_imag2.fit(data[1][:, 1, :, :, :], epochs = 1, batch_size = 16)
        
        vae_real3.fit(data[2][:, 0, :, :, :], epochs = 1, batch_size = 16)
        vae_imag3.fit(data[2][:, 1, :, :, :], epochs = 1, batch_size = 16)
        
    # train_loss.append(tracker.history['loss'])
    if ep % 10 == 0 and save_models:
        vae_encoder_imag1.save(f'AI_models/vae_enc_img1_ep{ep}') 
        vae_decoder_imag1.save(f'AI_models/vae_dec_img1_ep{ep}') 
        
        vae_encoder_real1.save(f'AI_models/vae_enc_rl1_ep{ep}') 
        vae_decoder_real1.save(f'AI_models/vae_dec_rl1_ep{ep}')
        
        
        vae_encoder_imag2.save(f'AI_models/vae_enc_img2_ep{ep}') 
        vae_decoder_imag2.save(f'AI_models/vae_dec_img2_ep{ep}') 
        
        vae_encoder_real2.save(f'AI_models/vae_enc_rl2_ep{ep}') 
        vae_decoder_real2.save(f'AI_models/vae_dec_rl2_ep{ep}')
        
        
        vae_encoder_imag3.save(f'AI_models/vae_enc_img3_ep{ep}') 
        vae_decoder_imag3.save(f'AI_models/vae_dec_img3_ep{ep}') 
        
        vae_encoder_real3.save(f'AI_models/vae_enc_rl3_ep{ep}') 
        vae_decoder_real3.save(f'AI_models/vae_dec_rl3_ep{ep}')
            

for i in range(3):
    print(f'models on data of cluster {i+1}')
    _,_,z_rl1 = vae_encoder_imag1.predict(data[i][:, 0, :, :, :])
    _,_,z_im1 = vae_encoder_imag1.predict(data[i][:, 1, :, :, :])
    
    _,_,z_rl2 = vae_encoder_imag2.predict(data[i][:, 0, :, :, :])
    _,_,z_im2 = vae_encoder_imag2.predict(data[i][:, 1, :, :, :])
    
    _,_,z_rl3 = vae_encoder_imag3.predict(data[i][:, 0, :, :, :])
    _,_,z_im3 = vae_encoder_imag3.predict(data[i][:, 1, :, :, :])
    
    
    
    out_rl1 = vae_decoder_imag1.predict(z_rl1)
    out_im1 = vae_decoder_imag1.predict(z_im1)

    out_rl2 = vae_decoder_imag2.predict(z_rl2)
    out_im2 = vae_decoder_imag2.predict(z_im2)
    
    out_rl3 = vae_decoder_imag3.predict(z_rl3)
    out_im3 = vae_decoder_imag3.predict(z_im3)

    x_true_rl1 = data[i][:, 0, :, :, :].reshape(-1, 6400)
    x_true_im1 = data[i][:, 1, :, :, :].reshape(-1, 6400)
    x_pred_rl1 = out_rl1.reshape(-1, 6400)
    x_pred_im1 = out_im1.reshape(-1, 6400)
    mse1 =( mean_squared_error(x_true_rl1, x_pred_rl1) +  mean_squared_error(x_true_im1, x_pred_im1)) / 2

    x_true_rl2 = data[i][:, 0, :, :, :].reshape(-1, 6400)
    x_true_im2 = data[i][:, 1, :, :, :].reshape(-1, 6400)
    x_pred_rl2 = out_rl2.reshape(-1, 6400)
    x_pred_im2 = out_im2.reshape(-1, 6400)
    mse2 =( mean_squared_error(x_true_rl2, x_pred_rl2) +  mean_squared_error(x_true_im2, x_pred_im2)) / 2

    x_true_rl3 = data[i][:, 0, :, :, :].reshape(-1, 6400)
    x_true_im3 = data[i][:, 1, :, :, :].reshape(-1, 6400)
    x_pred_rl3 = out_rl3.reshape(-1, 6400)
    x_pred_im3 = out_im3.reshape(-1, 6400)
    mse3 =( mean_squared_error(x_true_rl3, x_pred_rl3) +  mean_squared_error(x_true_im3, x_pred_im3)) / 2

    print(f'Model of cluster 1 shows total MSE loss to be {mse1}')
    print(f'Model of cluster 1 shows total MSE loss to be {mse2}')
    print(f'Model of cluster 1 shows total MSE loss to be {mse3}')