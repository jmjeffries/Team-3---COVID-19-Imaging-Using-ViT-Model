import tensorflow as tf
import keras
from keras import layers
from keras import ops
import numpy as np
from patches import Patches
from patchencoder import PatchEncoder

def get_model():
    model = keras.saving.load_model('model72.keras', custom_objects={"Patches": Patches, "PatchEncoder":PatchEncoder})
    return model
