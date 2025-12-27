#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BORROWING THE RIZZ - EXERCISE 1: FINE-TUNING THE PRO BRAIN
Now we're unfreezing the last 20 layers of MobileNetV2 to let it adjust its stance a little.
Think of this like:
- You (JS Dev) = Still a noob, but now Kelly Slater is giving you **personalized tips**.
- MobileNetV2 = Kelly Slater, but now he's **watching your footage and tweaking your form**.
"""

# 1. THE SETUP (Import the shit we need)
import tensorflow as tf  # The AI brain framework (like React for AI)
from tensorflow.keras import layers, models  # Tools to build the AI brain (like components in React)
import numpy as np  # Math stuff (like using lodash for numbers)
import time  # Time tracking (like setTimeout in JS, but for Python)

# 2. DOWNLOAD THE "PRO BRAIN" (MobileNetV2)
# We say include_top=False to chop off its original "head" (output layer)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),  # Shape of the images (224x224 pixels, 3 colors = RGB)
    include_top=False,  # We don't want its original brain, just the body
    weights='imagenet'  # Pre-trained weights (like Kelly's muscle memory)
)

# FREEZE THE BASE MODEL (so it doesn't forget Kelly's skills)
# This is like putting Kelly Slater in a straightjacket... but then loosening it a little.
base_model.trainable = True  # Unfreeze the entire model first (like taking off the straightjacket)
for layer in base_model.layers[:-20]:  # Freeze all layers except the last 20
    layer.trainable = False  # Now Kelly can only adjust your stance, not your entire body

# 3. BUILD YOUR OWN "SHITTY BRAIN" ON TOP
model = models.Sequential([
    base_model,  # The pro brain (Kelly Slater)
    layers.GlobalAveragePooling2D(),  # Squish the 3D data into 1D (like flattening a surfboard into a pancake)
    layers.Dense(128, activation='relu'),  # A layer with 128 neurons (like adding 128 tiny surf instructors to your brain)
    layers.Dropout(0.2),  # Randomly turn off 20% of neurons (like randomly forgetting 20% of Kelly's tips)
    layers.Dense(10, activation='softmax')  # Final layer for 10 classes (like choosing between 10 different surf tricks)
])

# 4. COMPILE THE MODEL (Get it ready for training)
# Lower learning rate because we're fine-tuning (like taking smaller steps when learning a new trick)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Smaller steps = less chance of wiping out
    loss='sparse_categorical_crossentropy',  # The loss function (like a scoreboard for your surfing)
    metrics=['accuracy']  # What we track (like counting how many waves you don't wipe out on)
)

# 5. LOAD YOUR "SHITTY DATASET" (CIFAR-10)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = tf.image.resize(x_train, (224, 224))  # Stretch training images
x_test = tf.image.resize(x_test, (224, 224))  # Stretch test images
x_train = x_train / 255.0  # Normalize (0-255 → 0-1)
x_test = x_test / 255.0  # Normalize (0-255 → 0-1)

# 6. TRAIN THE MODEL (Start copying Kelly's moves)
print("🏄‍♂️ Starting fine-tuning... (Kelly Slater is watching your footage and adjusting your stance)")
start_time = time.time()  # Start the timer

history = model.fit(
    x_train, y_train,  # Your shitty dataset (training data)
    epochs=5,  # Number of training cycles (like 5 days at surf camp)
    validation_data=(x_test, y_test)  # Test data (like your first real surf session)
)

# Stop the timer and print how long it took
training_time = time.time() - start_time
print(f"🕒 Fine-tuning took {training_time:.2f} seconds (or {training_time/60:.2f} minutes)")
print("🎉 Fine-tuning complete! Now go shred some waves, dude!")

# 7. EVALUATE THE MODEL (See if Kelly's tips helped)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n📊 Test accuracy: {test_acc:.4f} (If this is below 0.5, you're still cooked)")

# 8. SAVE THE MODEL (So you don't have to train it again)
model.save('borrowed_rizz_finetuned.h5')
print("💾 Fine-tuned model saved as 'borrowed_rizz_finetuned.h5' (Now you can pretend you're almost Kelly Slater)")