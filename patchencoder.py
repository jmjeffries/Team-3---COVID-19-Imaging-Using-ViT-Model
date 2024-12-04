import keras
from keras import ops
from keras import layers

@keras.saving.register_keras_serializable(package="Custom")
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):  
        super().__init__(**kwargs)  
        self.num_patches = num_patches
        self.projection_dim = projection_dim  # save projection_dim
        print(f'num_patches: {num_patches}, proj. dim: {projection_dim}')
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def build(self, input_shape):  
        super().build(input_shape)

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        config.update({"projection_dim": self.projection_dim})  # this line
        return config