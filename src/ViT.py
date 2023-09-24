from typing import Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from keras import layers
import matplotlib.pyplot as plt
from src.BuildingBlocks import *

class ViT_classifier:

    def __init__(self, x_train, params):

        self.learning_rate = params["learning_rate"]
        self.weight_decay = params["weight_decay"]
        self.batch_size = params["batch_size"]
        self.num_epochs = params["num_epochs"]
        self.input_shape = params["input_shape"]
        self.num_classes = params["num_classes"]
        self.image_size = params["image_size"]
        self.patch_size = params["patch_size"]
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection_dim = params["projection_dim"]
        self.num_heads = params["num_heads"]
        self.transformer_units = params["transformer_units"]
        self.transformer_layers = params["transformer_layers"]
        self.mlp_head_units = params["mlp_head_units"]

    def _build_model(self,x_train):
        # Input data.
        inputs = layers.Input(shape=self.input_shape)
        # Augment data.
        augmented = DataAugmentation(self.image_size, x_train)(inputs)
        # Create patches.
        patches= Patches(self.patch_size)(augmented)
        # Encode patches.
        encoded_patches= PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # Multiple Transformer Blocks.
        for _ in range(self.transformer_layers):
            encoded_patches = TransformerBlock(self.num_heads, self.projection_dim, self.transformer_units)(x = encoded_patches)

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)

        # Add MLP.
        features = MLP(representation, hidden_units = self.mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(self.num_classes)(features)
        # Create the Keras model.
        self.model = keras.Model(inputs=inputs, outputs=logits)

    def __call__(self, x):
        return self.model(x)
    
    def train(self, x_train, x_test, y_train, y_test):

        # Create the Keras model.
        self._build_model(x_train)

        optimizer = tfa.optimizers.AdamW(
            learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )

        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )

        checkpoint_filepath = "/tmp/checkpoint"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

        history = self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            validation_split=0.1,
            callbacks=[checkpoint_callback],
        )

        self.model.load_weights(checkpoint_filepath)
        _, accuracy, top_5_accuracy = self.model.evaluate(x_test, y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

        return history