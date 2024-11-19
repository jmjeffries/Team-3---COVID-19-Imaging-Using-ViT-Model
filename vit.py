import tensorflow as tf
import keras
from keras import layers
from keras import ops
import numpy as np

def get_model():
    model = keras.saving.load_model("model.keras", custom_objects='Patches''PatchEncoder')
    x = keras.random.uniform((72, 72, 1))
    assert np.allclose(model.predict(x), model.predict(x))
    return model

get_model()