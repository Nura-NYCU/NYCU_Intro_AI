from AI_models import classification
import numpy as np
import os
import random
random.seed(42) 
import tensorflow as tf
import keras
from sklearn.preprocessing import LabelBinarizer

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

accuracy = []

binarizer = LabelBinarizer()
binarizer.fit([0,1,2])

for ep in range(71):
    if ep % 10 == 0 and save_models:
        vae_encoder_imag1 = keras.models.load_model(f'AI_models/vae_enc_img1_ep{ep}') 
        vae_encoder_real1 = keras.models.load_model(f'AI_models/vae_enc_rl1_ep{ep}') 
        vae_encoder_imag2 = keras.models.load_model(f'AI_models/vae_enc_img2_ep{ep}') 
        vae_encoder_real2 = keras.models.load_model(f'AI_models/vae_enc_rl2_ep{ep}')
        vae_encoder_imag3 = keras.models.load_model(f'AI_models/vae_enc_img3_ep{ep}')
        vae_encoder_real3 = keras.models.load_model(f'AI_models/vae_enc_rl3_ep{ep}') 
        
    
        inp_data = []
        labels = []
        c = 0
        for i in range(10000):
            if c == 0:
                inp_data.append(data[0][i])
                labels.append(binarizer.transform([0])[0])
                c+=1
            elif c == 1:
                inp_data.append(data[1][i])
                labels.append(binarizer.transform([1])[0])
                c+=1
            elif c == 2:
                inp_data.append(data[2][i])
                labels.append(binarizer.transform([2])[0])
                c=0
        inp_data = np.asarray(inp_data)
        labels = np.asarray(labels)
        print(f'Epoch {ep}, length of dataset {inp_data.shape[0]}')
        
        mean_rl1, vae_rl1, z_rl1 = vae_encoder_real1.predict(inp_data[:, 0, :, :, :])
        mean_im1, vae_im1, z_im1 = vae_encoder_imag1.predict(inp_data[:, 1, :, :, :])
        
        mean_rl2, vae_rl2, z_rl2 = vae_encoder_real2.predict(inp_data[:, 0, :, :, :])
        mean_im2, vae_im2, z_im2 = vae_encoder_imag2.predict(inp_data[:, 1, :, :, :])
        
        mean_rl3, vae_rl3, z_rl3 = vae_encoder_real3.predict(inp_data[:, 0, :, :, :])
        mean_im3, vae_im3, z_im3 = vae_encoder_imag3.predict(inp_data[:, 1, :, :, :])
        
        data_post = []
        for i in range(len(mean_rl1)):
            data1 = [np.mean(mean_rl1[i]), np.mean(vae_rl1[i]), np.mean(z_rl1[i]), np.var(z_rl1[i]), np.mean(mean_im1[i]), np.mean(vae_im1[i]), np.mean(z_im1[i]), np.var(z_im1[i])]
            data2 = [np.mean(mean_rl2[i]), np.mean(vae_rl2[i]), np.mean(z_rl2[i]), np.var(z_rl2[i]), np.mean(mean_im2[i]), np.mean(vae_im2[i]), np.mean(z_im2[i]), np.var(z_im2[i])]
            data3 = [np.mean(mean_rl3[i]), np.mean(vae_rl3[i]), np.mean(z_rl3[i]), np.var(z_rl3[i]), np.mean(mean_im3[i]), np.mean(vae_im3[i]), np.mean(z_im3[i]), np.var(z_im3[i])]
            
            data_post.append(np.array([[data1, data2, data3]]).flatten())

        data_post = np.asarray(data_post)

        model = classification()
        with tf.device('/GPU:0'):
            tracker = model.fit(data_post, labels, epochs = 30, batch_size = 32, validation_split = 0.1)
        
        accuracy.append(tracker.history['accuracy'])
        model.save(f'AI_models/classification_ep{ep}')
