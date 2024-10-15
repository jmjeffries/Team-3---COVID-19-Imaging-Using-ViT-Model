import tensorflow as tf
import keras
from keras import layers
from keras import ops

import numpy as np
import matplotlib.pyplot as plt

import os
import glob
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf

import os

os.environ["KERAS_BACKEND"] = "jax"  # @param ["tensorflow", "jax", "torch"]

# Function to rename image files in a folder
def rename_image_files(folderpath, prefix, file_extension):
    for i, file_path in enumerate(glob.glob(os.path.join(folderpath, f'*.{file_extension}'))):
        new_file_name = f"{prefix}_{i+1}.{file_extension}"
        os.rename(file_path, os.path.join(folderpath, new_file_name))

# Function to process images and create a DataFrame with features and labels
def process_images(img_dir_path, binary_label, img_size=(72, 72)):
    img_names = [entry.name for entry in os.scandir(img_dir_path)]
    all_features = []

    for img in img_names:
        path = os.path.join(img_dir_path, img)
        cv_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if cv_img is not None:  # Ensure the image is loaded
            cv_img_resized = cv2.resize(cv_img, img_size, interpolation=cv2.INTER_NEAREST)
            features = np.reshape(cv_img_resized, (img_size[0] * img_size[1]))
            all_features.append(features)

    imgs_df = pd.DataFrame(np.array(all_features), index=img_names)
    imgs_df['class_label'] = np.ones((imgs_df.shape[0]), dtype=int) if binary_label else np.zeros((imgs_df.shape[0]), dtype=int)
    return imgs_df

# Function to shuffle dataset
def shuffle_dataset(df, shuffle_count=100):
    for _ in range(shuffle_count):
        df = df.sample(frac=1).reset_index(drop=True)
    return df

# Function to preprocess data: encode labels and prepare input-output data
def preprocess_data(df, img_size=(72, 72)):
    # Check if the number of features matches the expected size
    expected_num_features = img_size[0] * img_size[1]
    actual_num_features = df.shape[1] - 1  # Minus 1 for the label column

    if actual_num_features != expected_num_features:
        raise ValueError(f"Expected {expected_num_features} features, but got {actual_num_features} features. Check the input data.")

    # Encoding labels
    label_encoder = preprocessing.LabelEncoder()
    df['output_encode'] = label_encoder.fit_transform(df['class_label'])
    df = pd.get_dummies(df, columns=['output_encode'])

    #display(df)
    # Reshape the input data
    input_data_x = df.iloc[:, :-3].to_numpy().reshape((-1, img_size[0], img_size[1], 1))  # Reshape features for CNN
    output_label_y = df[['output_encode_0', 'output_encode_1']].to_numpy()  # One-hot encoded labels
    return input_data_x, output_label_y

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 2  # For real training, use num_epochs=100. 10 is a test value
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [
    2048,
    1024,
]  # Size of the dense layers of the final classifier


# Main workflow
def main():
    # Define paths and prefixes
    path1 = 'D:/Coding/Senior Project/FinalYear_Student_Project/CT_COVID'
    prefix1 = 'ct_covid'
    path2 = 'D:/Coding/Senior Project/FinalYear_Student_Project/CT_NonCOVID'
    prefix2 = 'ct_noncovid'

    # Uncomment if you need to rename images
    # rename_image_files(path1, prefix1, 'png')  # Example: rename PNG images
    # rename_image_files(path2, prefix2, 'png')

    # Process images and generate features
    ct_covid_features_df = process_images(path1, 1)  # 1 for covid-19 positive
    ct_noncovid_features_df = process_images(path2, 0)  # 0 for non-covid

    # Combine datasets
    combined_df = pd.concat([ct_covid_features_df, ct_noncovid_features_df])
    combined_df = shuffle_dataset(combined_df)

    # Preprocess data
    input_data_x, output_label_y = preprocess_data(combined_df)

#     print(f'Input_x Data Shape: {input_data_x.shape}')
#     print(f'Output_y Data Shape: {output_label_y.shape}')

    # Split data into train and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        input_data_x, output_label_y, test_size=0.2, random_state=42)

    return train_features, test_features, train_labels, test_labels

if __name__ == "__main__":
    train_features, test_features, train_labels, test_labels = main()


data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(train_features)

@tf.keras.utils.register_keras_serializable()
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

@tf.keras.utils.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

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
        return config

num_classes = 2
input_shape = (72, 72, 1)


def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

model = create_vit_classifier()
model.load_weights('model_weights.weights.h5')

optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

test = model.evaluate(test_features, test_labels, verbose=1)
print(f"Test Loss: {test[0]}")
print(f"Test Accuracy: {test[1]}")

def get_model():
    return model