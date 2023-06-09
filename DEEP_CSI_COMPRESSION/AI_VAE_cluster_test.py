import numpy as np
import os
import random
random.seed(42) 
import keras
from sklearn.metrics import mean_squared_error

'load data'
path = 'DIS_lab_LoS/samples/'

filenames = os.listdir(path)
random.shuffle(filenames)

# Artificial clustering
# Cluster 1 --> L2 norm in range 38-44
# Cluster 2 --> L2 norm in range 48-54
# Cluster 3 --> L2 norm in range 58-64
max_size = 200
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

accuracy = []

mse_tot = []
for ep in range(71):
    if ep % 10 == 0:
        vae_encoder_imag1 = keras.models.load_model(f'AI_models/vae_enc_img1_ep{ep}') 
        vae_encoder_real1 = keras.models.load_model(f'AI_models/vae_enc_rl1_ep{ep}') 
        vae_encoder_imag2 = keras.models.load_model(f'AI_models/vae_enc_img2_ep{ep}') 
        vae_encoder_real2 = keras.models.load_model(f'AI_models/vae_enc_rl2_ep{ep}')
        vae_encoder_imag3 = keras.models.load_model(f'AI_models/vae_enc_img3_ep{ep}')
        vae_encoder_real3 = keras.models.load_model(f'AI_models/vae_enc_rl3_ep{ep}') 
        
        vae_decoder_imag1 = keras.models.load_model(f'AI_models/vae_dec_img1_ep{ep}') 
        vae_decoder_real1 = keras.models.load_model(f'AI_models/vae_dec_rl1_ep{ep}') 
        vae_decoder_imag2 = keras.models.load_model(f'AI_models/vae_dec_img2_ep{ep}') 
        vae_decoder_real2 = keras.models.load_model(f'AI_models/vae_dec_rl2_ep{ep}')
        vae_decoder_imag3 = keras.models.load_model(f'AI_models/vae_dec_img3_ep{ep}')
        vae_decoder_real3 = keras.models.load_model(f'AI_models/vae_dec_rl3_ep{ep}') 
    
        model = keras.models.load_model(f'AI_models/classification_ep{ep}')
    
        inp_data = []
        labels = []
        c = 0
        for i in range(len(data[0])):
            if c == 0:
                inp_data.append(data[0][i])
                labels.append(0)
                c+=1
            elif c == 1:
                inp_data.append(data[1][i])
                labels.append(1)
                c+=1
            elif c == 2:
                inp_data.append(data[2][i])
                labels.append(2)
                c=0
        inp_data = np.asarray(inp_data)
        labels = np.asarray(labels)
        print(f'Epoch {ep}, length of dataset {inp_data.shape[0]}')
        
        acc_classification = 0
        mse = 0
        for label, csi in zip(labels, inp_data):
            mean_rl1, vae_rl1, z_rl1 = vae_encoder_real1.predict(csi[0, :, :, :].reshape(-1, 64, 100, 1))
            mean_im1, vae_im1, z_im1 = vae_encoder_imag1.predict(csi[1, :, :, :].reshape(-1, 64, 100, 1))
            
            mean_rl2, vae_rl2, z_rl2 = vae_encoder_real2.predict(csi[0, :, :, :].reshape(-1, 64, 100, 1))
            mean_im2, vae_im2, z_im2 = vae_encoder_imag2.predict(csi[1, :, :, :].reshape(-1, 64, 100, 1))
            
            mean_rl3, vae_rl3, z_rl3 = vae_encoder_real3.predict(csi[0, :, :, :].reshape(-1, 64, 100, 1))
            mean_im3, vae_im3, z_im3 = vae_encoder_imag3.predict(csi[1, :, :, :].reshape(-1, 64, 100, 1))
            
            data_post = []

            data1 = [np.mean(mean_rl1), np.mean(vae_rl1), np.mean(z_rl1), np.var(z_rl1), np.mean(mean_im1), np.mean(vae_im1), np.mean(z_im1), np.var(z_im1)]
            data2 = [np.mean(mean_rl2), np.mean(vae_rl2), np.mean(z_rl2), np.var(z_rl2), np.mean(mean_im2), np.mean(vae_im2), np.mean(z_im2), np.var(z_im2)]
            data3 = [np.mean(mean_rl3), np.mean(vae_rl3), np.mean(z_rl3), np.var(z_rl3), np.mean(mean_im3), np.mean(vae_im3), np.mean(z_im3), np.var(z_im3)]
            
            data_post.append(np.array([[data1, data2, data3]]).flatten())

            data_post = np.asarray(data_post)

            pred_classification = model.predict(data_post)
            clas = np.argmax(pred_classification)
            if clas == label:
                acc_classification += 1
                
            # decode
            if clas == 0:
                out_rl = vae_decoder_real1.predict(z_rl1)
                out_im = vae_decoder_imag1.predict(z_im1)
            
            elif clas == 1:
                out_rl = vae_decoder_real2.predict(z_rl2)
                out_im = vae_decoder_imag2.predict(z_im1)
              
            elif clas == 2:
                out_rl = vae_decoder_real3.predict(z_rl3)
                out_im = vae_decoder_imag3.predict(z_im3)
            
            x_true_rl = csi[0, :, :, :].reshape(-1, 6400)
            x_true_im = csi[1, :, :, :].reshape(-1, 6400)
                
            x_pred_rl = out_rl.reshape(-1, 6400)
            x_pred_im = out_im.reshape(-1, 6400)
            
            mse_tmp = (mean_squared_error(x_true_rl, x_pred_rl) + mean_squared_error(x_true_im, x_pred_im)) / 2
            mse += mse_tmp
        mse_tot.append(mse/len(labels))
        print(f'accuracy of encoder selection is {acc_classification / len(labels)}')
        accuracy.append(acc_classification / len(labels))

import matplotlib.pyplot as plt
plt.figure()
plt.title('Accuracy of selecting right encoder')
plt.plot(accuracy)
plt.xlabel('Epochs')
plt.xticks(np.arange(0,10), [0,10,20,30,40,50,60,70,80, 90])
plt.xlim(0,7)
plt.ylabel('accuracy')


plt.figure()
plt.title('Loss of Decoder at BS')
plt.plot(mse_tot)
plt.xlabel('Epochs')
plt.xticks(np.arange(0,10), [0,10,20,30,40,50,60,70,80, 90])
plt.xlim(0,7)
plt.ylabel('MSE')

np.save('SN_accuracy.npy', np.array(accuracy))
np.save('SN_Loss.npy', np.array(mse_tot))