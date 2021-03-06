import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.layers import Activation,BatchNormalization,Flatten,MaxPooling2D,ZeroPadding2D
from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose,Dense,Reshape,Dropout,UpSampling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from scipy import misc
import numpy as np
import os
import numpy as np



class CEncoder():
    def __init__(self,batch_size,dir_path,cpkt_period,path_discriminator,path_generator):
        self.img_rows = 128
        self.img_cols = 128
        self.mask_height = 64
        self.mask_width = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.patch_shape = (self.mask_height,self.mask_width,self.channels)
        self.overlap = 7
        self.batch_size=batch_size
        self.path_discriminator=path_discriminator
        self.path_generator=path_generator

        self.ckpt_period = cpkt_period

        self.dir_path=dir_path

        lambda_recon = 0.999
        lambda_adv = 0.001
        optimizer_recon = Adam(lr=2e-3, beta_1=0.5,decay=1e-2)
        optimizer_adv = Adam(lr=2e-3, beta_1=0.5,decay=1e-3)

        self.gf = 64
        self.df = 64 # masked patch size

        # Construct autoencoder
        self.autoencoder = self.build_autoencoder()
        # Construct discriminator
        self.discriminator = self.build_discriminator()
        
        #Load weights 
        if(path_generator!=""):
            self.autoencoder.load_weights(dir_path+"/"+path_generator)
        if(path_discriminator!=""):
            self.discriminator.load_weights(dir_path+"/"+path_discriminator)
        

        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer_adv, metrics=['accuracy'])
        # Construct GAN
        masked_img = Input(shape=self.img_shape)
        #gen_missing = self.autoencoder(masked_img)
        gen_missing = self.autoencoder(masked_img)

        self.discriminator.trainable = True

        valid = self.discriminator(gen_missing)

        self.combined = Model(masked_img, [gen_missing, valid])
        self.combined.compile(loss=['mse','binary_crossentropy'],
            loss_weights=[lambda_recon,lambda_adv],
            optimizer=optimizer_recon,
            metrics=['accuracy'])

    """ Building models """
    def build_autoencoder(self):
        def conv_bn_relu_pool(input_layer,filters,kernel_size=4,stride=2,pad='same',activation='relu'):
            y = Conv2D(filters,kernel_size=kernel_size,
                                strides=stride,padding=pad)(input_layer)

            if activation=='relu':
                y = Activation('relu')(y)
            elif activation=='lrelu':
                y = tf.keras.layers.LeakyReLU(alpha=0.2)(y)
            else:
                print("WARNING: no activation layer")
                pass
            y = BatchNormalization()(y)
            output = y
            return output
        def deconv_bn_relu(input_layer,filters,kernel_size=4,stride=2,pad='same', activation='relu'):
            y = Conv2DTranspose(filters,kernel_size=kernel_size,
                                strides=stride,padding=pad)(input_layer)
            if activation=='relu':
                y = Activation('relu')(y)
            elif activation=='lrelu':
                y = tf.keras.layers.LeakyReLU(alpha=0.2)(y)
            else:
                input("WARNING: no activation layer <Enter to continue>")
                pass
            y = BatchNormalization()(y)
            output = y
            return output


        # AlexNet
        input = Input(shape=self.img_shape)

        y = conv_bn_relu_pool(input,filters=self.gf,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.gf,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.gf*2,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.gf*4,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.gf*8,activation='relu')
        latent = conv_bn_relu_pool(y,filters=4000,stride=1,pad='valid',activation='lrelu') # increase to 4000
        latent = Dropout(0.5)(latent)

        y = deconv_bn_relu(latent,filters=self.df*8,stride=1,pad='valid',activation='relu')
        y = deconv_bn_relu(y,filters=self.df*4,activation='relu')
        y = deconv_bn_relu(y,filters=self.df*2,activation='relu')
        y = deconv_bn_relu(y,filters=self.df,activation='relu')

        y = Conv2DTranspose(3,kernel_size=4,strides=2,padding='same')(y)
        output = Activation('tanh')(y)

        model = Model(input,output)
        model.summary()
        return model

    def build_discriminator(self):
        def conv_bn_relu_pool(input_layer,filters,kernel_size=4,stride=2,pad='same',activation='relu'):
            y = Conv2D(filters,kernel_size=kernel_size,
                        strides=stride,padding=pad)(input_layer)
            if activation=='relu':
                y = Activation('relu')(y)
            elif activation=='lrelu':
                y = tf.keras.layers.LeakyReLU(alpha=0.2)(y)
            else:
                print("WARNING: no activation layer")
                pass
            y = BatchNormalization()(y)
            output = y
            return output

        patch = Input(shape=self.patch_shape)
        y = conv_bn_relu_pool(patch,filters=self.df,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.df*2,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.df*4,activation='lrelu')
        y = conv_bn_relu_pool(y,filters=self.df*8,activation='lrelu')

        y = Flatten()(y)
        validity = Dense(1,activation='sigmoid')(y) # review activation here
        model = Model(patch,validity)

        model.summary()
        return model

    
    
    def train(self, generator,max_iter=1,start_iter=0):
        
        step_counter=start_iter
        
        half_batch = int(self.batch_size / 2)
        writer = tf.summary.create_file_writer(self.dir_path)

        while(step_counter<max_iter):
            #for _, x_batch_train in enumerate(generator):
            generator.on_epoch_end()
            
            for i in range(0,len(generator),2):
                step_counter+=1

                # ---------------------
                #  Train Discriminator
                # ---------------------

                x_batch_train = generator.__getitem__(i)
                imgs,masked,missings = x_batch_train

                idx = np.random.choice(range(self.batch_size), size=half_batch, replace=False, p=None)
    
                half_masked = np.array([masked[i] for i in idx])
                half_missing = np.array([missings[i] for i in idx])
                
    
                gen_missing = self.autoencoder.predict(half_masked)
                valid = np.ones((half_batch, 1))
                fake = np.zeros((half_batch, 1))
                
                d_loss_real = self.discriminator.train_on_batch(half_missing, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_missing, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # ---------------------
                #  Train Generator
                # ---------------------

                x_batch_train = generator.__getitem__(i+1)
                imgs,masked,missings = x_batch_train
              
                valid = np.ones((self.batch_size, 1))

                # Train the generator
                g_loss = self.combined.train_on_batch(masked, [missings, valid])
                
                
                # Plot the progress
                print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f, acc: %.2f%%]" % (step_counter, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1], 100*g_loss[4]))

                if step_counter%self.ckpt_period ==0:
                    self.autoencoder.save_weights(self.dir_path+"/ckpt_context_generator"+str(step_counter))
                    self.discriminator.save_weights(self.dir_path+"/ckpt_context_discriminator"+str(step_counter))

                with writer.as_default():
                    tf.summary.scalar('d_loss', d_loss[0], step=step_counter)
                    tf.summary.scalar('discriminator_accuracy', 100*d_loss[1], step=step_counter)
                    tf.summary.scalar('g_loss_discriminator',g_loss[0] , step=step_counter)
                    tf.summary.scalar('g_loss_mse',g_loss[0] , step=step_counter)
                    tf.summary.scalar('g_loss_accuracy',100*g_loss[4] , step=step_counter)

                    