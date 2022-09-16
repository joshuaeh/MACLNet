# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")
sys.path.append("../src/data/")

import tensorflow as tf
from dataloader import *
from network import *
from loss import *
from config import *
from utils import *
import logging
import os

def main():
    """ training WaferSegClassNet model 
    """
    logger = logging.getLogger(__name__)

    logger.info("Setting gpu strategy")
    gpus = tf.config.list_logical_devices('GPU')
    # communication_options = tf.distribute.experimental.CommunicationOptions(
    #     implementation=tf.distribute.experimental.CommunicationImplementation.AUTO
    # )
    # strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options)
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        logger.info("[Info] Creating Network")
        model = getModel()
        model.compile(optimizer = tf.keras.optimizers.Adam(INITIAL_LEARNING_RATE), loss = bceDiceLoss, metrics = [diceCoef])
    logger.info("[Info] Summary of model \n")
    logger.info(model.summary())

    logger.info("[Info] Getting DataLoader")
    trainGen, testGen = getDataLoader(batch_size=1)

    logger.info("[Info] Creating Dataset from Dataloader")
    logger.debug("[Debug] Get Data generator length")
    train_size = trainGen.__len__
    test_size = testGen.__len__

    logger.debug("[Debug] Create dataset from generator")
    train_ds = tf.data.Dataset.from_generator(trainGen, args=[train_size], output_types=(tf.float32, tf.float32, tf.float32))
    test_ds = tf.data.Dataset.from_generator(testGen, args=[test_size], output_types=(tf.float32, tf.float32, tf.float32))

    logger.debug("[Debug] Disable Autoshard")
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_ds = train_ds.with_options(options)
    test_ds = test_ds.with_options(options)

    logger.debug("[Debug] Distribute dataset")
    train_ds = strategy.experimental_distribute_dataset(train_ds)
    test_ds = strategy.experimental_distribute_dataset(test_ds)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(WEIGHTS_DIR, 'ACLNet_Best.h5'), monitor = 'val_diceCoef', mode="max", verbose = 1, save_best_only = True, save_weights_only = False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_diceCoef', mode="max", factor = 0.1, patience = 20, min_lr = 0.00001)
    ]
    model.fit(trainGen, validation_data = testGen, epochs = EPOCHS, verbose = 1, callbacks = callbacks)
    logger.info("[Info] Training Finished")

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, filename = os.path.join(LOG_DIR, 'app.log'), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w')

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    main()
