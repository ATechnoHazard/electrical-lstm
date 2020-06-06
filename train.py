# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 22:32:18 2020

@author: Tanmay Thakur
"""

import pickle

from model import get_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


X_train, y_train = pickle.load(open( "dict.pickle", "rb" ))

model = get_model(X_train)

model.compile(loss = 'mae', optimizer = Adam(lr = 1e-3))

cp_callbacks = ModelCheckpoint(filepath = "recurrent_model_initial.h5", monitor = "val_loss", mode = 'min', save_best_only = True, verbose = 1)

model.fit(X_train, y_train, epochs = 40, batch_size = 16, validation_split = 0.25, callbacks = [cp_callbacks])
