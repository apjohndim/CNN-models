# -*- coding: utf-8 -*-
"""
⣶⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠙⠿⣷⣶⣤⣤⣤⣤⣤⣴⣶⣶⣾⣿⣿⣿⣿⣿⣿⣿⣶⣦⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⠙⠻⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠛⠛⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⡶⠆⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠛⠛⠋⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣾⣿⡿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡿⠟⠋⠁⣰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣟⠻⠿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣏⠙⠲⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⠀⠉⠛⠷⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠈⠙⠿⣶⣄⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡗⠀⠀⠀⠀⠀⠀⠀⠈⠙⢷⣄⣀⣾⣿⣷
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣿⣿⣿⡇
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠀
⠀⠀⠀⠀⠀⠀⠀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠟⠛⠛⠉⠉⠙⠛⠛⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠒⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⢿⣿⣿⣿⡟⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢀⣿⣿⣿⣿⣿⣿⣿⡿⠟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⢿⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢸⣿⣿⣿⡿⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠳⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣼⠿⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀


Copyright: Ioannis D. Apostolopoulos
Convolutional Neural Network Models

COLLECTION:
    - Baseline VGG19
    - Feature Fusion VGG19
    - Feature Fusion - LSTM VGG19
    - Attending Feature Fusion VGG19 (Att-FF-VGG19)

# conda install -c anaconda graphviz
# conda install -c anaconda pydot

"""


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import regularizers
from tensorflow.keras.layers import multiply,Lambda
import tensorflow.keras.backend as K
import numpy as np
import pydot


def furnish_base_vgg19 (in_shape, tune='auto', classes=2): #tune = 0 is off the self, tune = 1 is scratch, auto is last conv trainable
    
    ###########################################################################
    #
    # The baseline VGG19 with option for fine-tune and classic Dense layers with Global Average Pooling
    #
    ###########################################################################


    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune == 20 or tune=='auto':   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[20:]:
            layer.trainable = True
    #base_model.summary()
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    
    print("[INFO] VGG19 Baseline Model Compiled!")
    return model


def furnish_ffvgg19 (in_shape, tune, classes):
    
    ###########################################################################
    #
    # The baseline Feature Fusion VGG19
    #
    ###########################################################################
    
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', 
                                                   input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[19:]:
        layer.trainable = True
    #base_model.summary()
    
    # early2 = layer_dict['block2_pool'].output 
    # #early2 = Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early2)
    # early2 = tf.keras.layers.BatchNormalization()(early2)
    # early2 = tf.keras.layers.Dropout(0.5)(early2)
    # early2= tf.keras.layers.GlobalAveragePooling2D()(early2)
        
    early3 = layer_dict['block3_pool'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3 = tf.keras.layers.BatchNormalization()(early3)
    early3 = tf.keras.layers.Dropout(0.5)(early3)
    early3= tf.keras.layers.GlobalAveragePooling2D()(early3)    
        
    early4 = layer_dict['block4_pool'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4 = tf.keras.layers.BatchNormalization()(early4)
    early4 = tf.keras.layers.Dropout(0.5)(early4)
    early4= tf.keras.layers.GlobalAveragePooling2D()(early4)     
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1)
    x = tf.keras.layers.concatenate([x1, early4, early3], axis=-1)  
    
    x = tf.keras.layers.Dense(1500, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='ffvgg19-base.png')
    print("[INFO] Model Compiled!")
    return model



def furnish_ffvgg19_lstm(in_shape, tune, classes):
    
    ###########################################################################
    #
    # The baseline Feature Fusion VGG19 with LSTM units at the top
    #
    ###########################################################################
    
    '''FFVGG19'''
    def ffvgg19(in_shape, tune, classes):
        base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', 
                                                       input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
        layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
        #base_model.summary()
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[19:]:
            layer.trainable = False
      
        early3 = layer_dict['block3_pool'].output   
        early3 = tf.keras.layers.BatchNormalization()(early3)
        early3 = tf.keras.layers.Dropout(0.5)(early3)
        early3= tf.keras.layers.GlobalAveragePooling2D()(early3)    
            
        early4 = layer_dict['block4_pool'].output   
        early4 = tf.keras.layers.BatchNormalization()(early4)
        early4 = tf.keras.layers.Dropout(0.5)(early4)
        early4= tf.keras.layers.GlobalAveragePooling2D()(early4)     
        
        x1 = layer_dict['block5_conv3'].output 
        x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
        x = tf.keras.layers.concatenate([x1, early4, early3], axis=-1)
        exodus = tf.keras.layers.Dense(500, activation="relu")(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=exodus)
        
        return model
    
    
    cnn_net = ffvgg19(in_shape, tune, classes)
    input_layer = tf.keras.layers.Input(shape=(3, in_shape[0], in_shape[1], in_shape[2]))
    lstm_ip_layer = tf.keras.layers.TimeDistributed(cnn_net)(input_layer)
    
    x = tf.keras.layers.LSTM(500)(lstm_ip_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.LSTM(200)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.LSTM(50)(x)
    output = tf.keras.layers.Dense(units=2,activation='softmax')(x)
    model = tf.keras.Model([input_layer],output)

    optimizer = 'Adam'
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def furnish_attention_ffvgg19 (in_shape, tune, classes):
    
    ###########################################################################
    #
    # The baseline Feature Fusion VGG19 with attention blocks and BD paths
    #
    ###########################################################################
    
    from tensorflow.keras.utils import plot_model
    def pay_attention(m_input,filters, name):
      # Based on https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age#Attention-Model
      pt_depth = filters
      bn_features = BatchNormalization()(m_input)

      attn = Conv2D(64, kernel_size=(1,1), padding='same', activation='relu')(bn_features)
      attn = Conv2D(16, kernel_size=(1,1), padding='same', activation='relu')(attn)
      attn = Conv2D(1, 
                    kernel_size=(1,1), 
                    padding='valid', 
                    activation='sigmoid',
                    name=name)(attn)
      up_c2_w = np.ones((1, 1, 1, pt_depth))
      up_c2 = Conv2D(pt_depth,
                     kernel_size=(1,1),
                     padding='same', 
                     activation='linear',
                     use_bias=False,
                     weights=[up_c2_w])
      up_c2.trainable = False
      attn = up_c2(attn)

      mask_features = multiply([attn, bn_features])
      gap_features = GlobalAveragePooling2D()(mask_features)
      gap_mask = GlobalAveragePooling2D()(attn)
      attn_gap = Lambda(lambda x: x[0]/x[1])([gap_features, gap_mask])

      return attn_gap
  
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', 
                                                   input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[19:]:
        layer.trainable = False
  
    f3 = layer_dict['block3_pool'].output   
    f3 = tf.keras.layers.BatchNormalization()(f3)
    f3 = tf.keras.layers.Dropout(0.5)(f3)
    f3= tf.keras.layers.GlobalAveragePooling2D()(f3)   
    
    f3b = layer_dict['block3_pool'].output
    f3b = pay_attention(f3b,256, 'Att_b3p')   
    
    
    f4 = layer_dict['block4_pool'].output   
    f4 = tf.keras.layers.BatchNormalization()(f4)
    f4 = tf.keras.layers.Dropout(0.5)(f4)
    f4= tf.keras.layers.GlobalAveragePooling2D()(f4)     
    
    f4b = layer_dict['block4_pool'].output
    f4b = pay_attention(f4b,512, 'Att_b4p')          
    
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.concatenate([x1, f3, f3b,f4, f4b], axis=-1)
          
    
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])  
    plot_model(model, to_file='ffvgg19-base.png')
    return model
    

    
def furnish_attention_vgg19 (in_shape, tune, classes):
    
    ###########################################################################
    #
    # The baseline VGG19 with attention blocks
    #
    ###########################################################################
    
    from tensorflow.keras.utils import plot_model
    def pay_attention(m_input,filters, name):
      # Based on https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age#Attention-Model
      pt_depth = filters
      bn_features = BatchNormalization()(m_input)

      attn = Conv2D(64, kernel_size=(1,1), padding='same', activation='relu')(bn_features)
      attn = Conv2D(16, kernel_size=(1,1), padding='same', activation='relu')(attn)
      attn = Conv2D(1, 
                    kernel_size=(1,1), 
                    padding='valid', 
                    activation='sigmoid',
                    name=name)(attn)
      up_c2_w = np.ones((1, 1, 1, pt_depth))
      up_c2 = Conv2D(pt_depth,
                     kernel_size=(1,1),
                     padding='same', 
                     activation='linear',
                     use_bias=False,
                     weights=[up_c2_w])
      up_c2.trainable = False
      attn = up_c2(attn)

      mask_features = multiply([attn, bn_features])
      gap_features = GlobalAveragePooling2D()(mask_features)
      gap_mask = GlobalAveragePooling2D()(attn)
      attn_gap = Lambda(lambda x: x[0]/x[1])([gap_features, gap_mask])

      return attn_gap
  

    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', 
                                                   input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[19:]:
        layer.trainable = False
  
    f1 = layer_dict['block1_pool'].output
    f1 = pay_attention(f1,64, 'Att_b1p')   
    
    
    f2 = layer_dict['block2_pool'].output
    f2 = pay_attention(f2,128, 'Att_b2p')         
  
  
    f3 = layer_dict['block3_pool'].output
    f3 = pay_attention(f3,256, 'Att_b3p')   
    
    
    f4 = layer_dict['block4_pool'].output
    f4 = pay_attention(f4,512, 'Att_b4p')          
    
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.concatenate([x1,f1,f2,f3,f4], axis=-1)
          
    
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])  
    plot_model(model, to_file='vgg19-attention.png')
    return model