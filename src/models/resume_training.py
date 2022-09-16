import os
import sys
print(os.getcwd())
sys.path.append("../src")
sys.path.append("../src/data/")
sys.path.append("../src/models/")

import tensorflow as tf
from dataloader import *
from network import *
from loss import *
from config import *
from utils import *
import logging
import os

from models.loss import bceDiceLoss, diceCoef
sys.path.append("/work/08940/joshuaeh/maverick2/miniconda3/lib/:/work/08940/joshuaeh/maverick2/miniconda3/lib/:/work/08940/joshuaeh/maverick2/miniconda3/lib/:/work/08940/joshuaeh/maverick2/miniconda3/envs/tf/lib/:/work/08940/joshuaeh/maverick2/miniconda3/envs/tf/lib/:/work/08940/joshuaeh/maverick2/miniconda3/lib/:/work/08940/joshuaeh/maverick2/miniconda3/envs/tf/lib/:/work/08940/joshuaeh/maverick2/miniconda3/envs/tf/lib/")

logging.basicConfig(level = logging.INFO, filename = os.path.join(LOG_DIR, 'app.log'), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w')

sys.stdout = LoggerWriter(logging.info)
sys.stderr = LoggerWriter(logging.error)

logger = logging.getLogger(__name__)

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    logger.info("[Info] Getting DataLoader")
    trainGen, testGen = getDataLoader(batch_size=BATCH_SIZE)

    logger.info("[Info] Creating Network") 
    model = tf.keras.models.load_model("../weights/ACLNet_Best.h5", custom_objects={"bceDiceLoss":bceDiceLoss, "diceCoef":diceCoef})

    model.compile(optimizer = tf.keras.optimizers.Adam(INITIAL_LEARNING_RATE), loss = bceDiceLoss, metrics = [diceCoef])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(WEIGHTS_DIR, 'ACLNet_Best.h5'), monitor = 'val_diceCoef', mode="max", verbose = 1, save_best_only = True, save_weights_only = False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_diceCoef', mode="max", factor = 0.1, patience = 20, min_lr = 0.00001)
    ]
    model.fit(trainGen, validation_data = testGen, epochs = EPOCHS, verbose = 1, callbacks = callbacks)
    logger.info("[Info] Training Finished")
