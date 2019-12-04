# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:40:18 2019

@author: cloh5
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.__version__

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import datetime
import time
from skimage import io
from sklearn.metrics import accuracy_score
from skimage.transform import rescale

#from IPython import display

subj='ob'

def process_images(path, num_images):
  images = []
  for i in range(num_images):
    # Load in the images
    image=io.imread('data_ob/'+path+'/%04d' % i+'.png')
    #image=rescale(image,0.7) #only for jap
    images.append(np.array(image))
    
  images=np.array(images)
  
#  v=10 ##white
#  h=255 ##white
#  v=-20 ##lady
#  h=-50 ##lady
#  v=-30 ##jap
#  h=-140 ##jap
  v=0
  h=0
  #image_rescaled = rescale(images, 0.5, anti_aliasing=False)
  #plt.imshow(images)
  images_cropped = images[:, 30+v:480+v, 495+h:945+h, :]
  

  #assert images_cropped.shape == (num_images, 450, 450, 3)
  
  images = images_cropped.reshape(images_cropped.shape[0], 450, 450, 3).astype('float32')
  images = (images - 127.5) / 127.5 # Normalize the images to [-1, 1]
  #images = (images - 0.5) / 0.5 #only for jap
  assert images.min() >= -1.0
  assert images.max() <= 1.0
  
  return images

tot_num=389
real = process_images('real', tot_num)
plt.imshow(real[0])

#Load in the fake images
deepfake = process_images('deepfakes', tot_num)
face2face = process_images('face2face', tot_num)
faceswap = process_images('faceswap', tot_num)
neuraltextures = process_images('neuraltextures', tot_num)

BUFFER_SIZE = tot_num
BATCH_SIZE = 8

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(real).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce an image from a seed 
#(random noise). Start with a Dense layer that takes this seed as input, then upsample several times 
#until you reach the desired image size of 720, 1280, 3. Notice the tf.keras.layers.LeakyReLU activation for 
#each layer, except the output layer which uses tanh.

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(25*25*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((25, 25, 256)))
    assert model.output_shape == (None, 25, 25, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 25, 25, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(3, 3), padding='same', use_bias=False))
    assert model.output_shape == (None, 75, 75, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(3, 3), padding='same', use_bias=False))
    assert model.output_shape == (None, 225, 225, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 450, 450, 3)

    return model

#Use the (as yet untrained) generator to create an image.
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

#plt.imshow(((np.array(generated_image[0, :, :, :]))*127.5 + 127.5).astype(int))
#plt.show()
#
#plt.imshow((real[0]*127.5 + 127.5).astype(int))
#plt.show()

#The discriminator is a CNN-based image classifier.
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[450, 450, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))#, activation='sigmoid'))

    return model

#Use the (as yet untrained) discriminator to classify the generated images as real or fake. 
#The model will be trained to output positive values for real images, and negative values for fake images.
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


#This method quantifies how well the discriminator is able to distinguish real images from fakes. 
#It compares the discriminator's predictions on real images to an array of 1s, and the 
#discriminator's predictions on fake (generated) images to an array of 0s.

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

#The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, 
#if the generator is performing well, the discriminator will classify the fake images as real (or 1). 
#Here, we will compare the discriminators decisions on the generated images to an array of 1s.

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Define our metrics
G_loss = tf.keras.metrics.Mean('gen_loss', dtype=tf.float32)
D_loss = tf.keras.metrics.Mean('disc_loss', dtype=tf.float32)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


EPOCHS = 500
noise_dim = 100
num_examples_to_generate = 1

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
    
    G_loss(gen_loss)
    D_loss(disc_loss)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)
    with train_summary_writer.as_default():
      tf.summary.scalar('g_loss', G_loss.result(), step=epoch)
      tf.summary.scalar('d_loss', D_loss.result(), step=epoch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 1 epochs
    if (epoch + 1) % 1 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    G_loss.reset_states()
    D_loss.reset_states()

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(2, 2, i+1)
      plt.imshow((np.array(predictions[i, :, :, :]) * 127.5 + 127.5).astype(int))
      plt.axis('off')

  plt.savefig('GANsave_'+subj+'/image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

#tensorboard --logdir logs/gradient_tape

##time
train(train_dataset, EPOCHS)

#!zip -r /content/checkpoints.zip /content/training_checkpoints/

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

generated_images = np.array(generator(tf.random.normal([10, noise_dim]), training=False))
for i in range(38):
  generated_images_next = np.array(generator(tf.random.normal([10, noise_dim]), training=False))
  generated_images = np.concatenate((generated_images, generated_images_next), axis=0)
generated_images.shape

def predictions(test):
  pred = []
  for i in range(test.shape[0]):
    pred.append(discriminator(test[i].reshape(1,450,450,3))[0,0])  
  return np.array(pred).flatten()

real_pred = predictions(real)
deepfake_pred = predictions(deepfake)
face2face_pred = predictions(face2face)
faceswap_pred = predictions(faceswap)
neuraltextures_pred = predictions(neuraltextures)
generated_pred = predictions(generated_images)


plt.figure(figsize=(12,6))
preds=[real_pred, deepfake_pred, face2face_pred, faceswap_pred, neuraltextures_pred, generated_pred]
labels=['real', 'deepfake', 'face2face', 'faceswap', 'neuraltextures', 'generated']
for i in range(5):
  plt.hist(preds[i], 40,label = labels[i], alpha=0.7)
plt.title('Discriminator Output (separated by techniques)')
plt.legend()
#plt.show()
plt.savefig(subj+'pred_separated.png')

fig,ax=plt.subplots(1,1,figsize=(16,6))

ax.boxplot(preds)
plt.xticks([1, 2, 3, 4, 5, 6], labels)
plt.title('Discriminator Output (separated by techniques)')
#plt.show()
plt.savefig(subj+'pred_separated_box.png')

plt.figure(figsize=(12,6))
fake_preds=np.concatenate((deepfake_pred, face2face_pred, faceswap_pred, neuraltextures_pred, generated_pred))
plt.hist(real_pred, 40,label = 'real', alpha=0.7, density=True)
plt.hist(fake_preds, 40,label = 'fake', alpha=0.7, density=True)
plt.title('Discriminator Output')
plt.legend()
#plt.show()
plt.savefig(subj+'pred.png')
real_pred.mean(), fake_preds.mean()
threshold = 8
real_output = np.array([1 if real_pred[i]>threshold else 0 for i in range(len(real_pred))])
fake_output = np.array([1 if fake_preds[i]>threshold else 0 for i in range(len(fake_preds))])

y_true = np.concatenate((np.ones(len(real_pred)), np.zeros(len(fake_preds))))
y_pred = np.concatenate((real_output, fake_output))

accuracy_score(y_true, y_pred)

def epoch_tune(epoch):
    checkpoint.restore('./training_checkpoints/ckpt-'+str(epoch))
    real_pred = predictions(real)
    deepfake_pred = predictions(deepfake)
    face2face_pred = predictions(face2face)
    faceswap_pred = predictions(faceswap)
    neuraltextures_pred = predictions(neuraltextures)
    fake_preds=np.concatenate((deepfake_pred, face2face_pred, faceswap_pred, neuraltextures_pred))
    threshold = (real_pred.mean() + fake_preds.mean())/2
    real_output = np.array([1 if real_pred[i]>threshold else 0 for i in range(len(real_pred))])
    fake_output = np.array([1 if fake_preds[i]>threshold else 0 for i in range(len(fake_preds))])
    y_true = np.concatenate((np.ones(len(real_pred)), np.zeros(len(fake_preds))))
    y_pred = np.concatenate((real_output, fake_output))
      
    hist_real = cv2.calcHist([real_pred], [0], None, [40], [-20,20])
    cv2.normalize(hist_real, hist_real)
    
    hist_fakes = cv2.calcHist([fake_preds], [0], None, [40], [-20,20])
    cv2.normalize(hist_fakes, hist_fakes)
    
    hist_real=hist_real/np.sum(hist_real)
    hist_fakes=hist_fakes/np.sum(hist_fakes)
    
    intsec=cv2.compareHist(hist_real,hist_fakes,cv2.HISTCMP_INTERSECT)
    correl=cv2.compareHist(hist_real,hist_fakes,cv2.HISTCMP_CORREL)
    KL1=cv2.compareHist(hist_real,hist_fakes,cv2.HISTCMP_KL_DIV)
    KL2=cv2.compareHist(hist_fakes,hist_real,cv2.HISTCMP_KL_DIV)
    bhat=cv2.compareHist(hist_real,hist_fakes,cv2.HISTCMP_BHATTACHARYYA)
    
    return accuracy_score(y_true, y_pred),intsec,correl,KL1,KL2,bhat

epochs = np.linspace(1, 500, 50, dtype='int')
#diff_values = []
acc_values = []
int_values = []
correl_values = []
KL1_values = []
KL2_values = []
bhat_values = []
for epoch in epochs:
    print(epoch)
    acc,intsecout,correlout,KL1out,KL2out,bhatout = epoch_tune(epoch)
    acc_values.append(acc)
    int_values.append(intsecout)
    correl_values.append(correlout)
    KL1_values.append(KL1out)
    KL2_values.append(KL2out)
    bhat_values.append(bhatout)
#plt.figure(figsize=(12,6))
#plt.plot(np.linspace(10, 500, 50, dtype='int'), diff_values, '-o')
#plt.xlabel('Epochs')
#plt.ylabel('Class Mean Separation')
#plt.title('Class mean separation during training')
##plt.show()
#plt.savefig('meansep_train_500.png')

plt.figure(figsize=(12,6))
plt.plot(np.linspace(1, 500, 50, dtype='int'), acc_values)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation accuracy during training')
#plt.show()
plt.savefig(subj+'acc_500.png')


plt.figure(figsize=(12,6))
plt.plot(np.linspace(1, 500, 50, dtype='int'), correl_values)
plt.xlabel('Epochs')
plt.ylabel('Correlation')
plt.title('Correlation during training')
#plt.show()
plt.savefig(subj+'correl_500.png')

plt.figure(figsize=(12,6))
plt.plot(np.linspace(1, 500, 50, dtype='int'), int_values)
plt.xlabel('Epochs')
plt.ylabel('Intersection')
plt.title('Intersection during training')
#plt.show()
plt.savefig(subj+'intsec_500.png')

plt.figure(figsize=(12,6))
plt.plot(np.linspace(1, 500, 50, dtype='int'), KL1_values,np.linspace(1, 500, 50, dtype='int'), KL2_values)
plt.xlabel('Epochs')
plt.ylabel('KL divergence')
plt.title('KL divergence during training')
#plt.show()
plt.savefig(subj+'KLdiv_500.png')

plt.figure(figsize=(12,6))
plt.plot(np.linspace(1, 500, 50, dtype='int'), bhat_values)
plt.xlabel('Epochs')
plt.ylabel('Bhattacharyya Distance')
plt.title('Bhattacharyya Distance during training')
#plt.show()
plt.savefig(subj+'bhat_500.png')