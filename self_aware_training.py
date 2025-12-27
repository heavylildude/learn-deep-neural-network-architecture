#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# """
# SELF-AWARE TRAINING (The "AI Brain Coach")
# ------------------------------------------
# This script teaches an AI brain to recognize handwritten numbers (0-9) using:
# - **Dropout**: Randomly turns off neurons to prevent "cheating" (like covering your eyes while studying).
# - **Callbacks**: Smart tools that watch training and stop early if the AI isn't learning.
# - **Validation Split**: A "test set" to check if the AI is actually getting smarter.
# 
# Think of this like training a surfer:
# - The AI starts as a kook (bad at recognizing numbers).
# - Dropout forces it to learn properly (no peeking!).
# - Callbacks act like a coach who stops training if the surfer isn't improving.
# """
# ### **Key Concepts Explained (Like You're 10)**
# 1. **Dropout**:
   # - Imagine you're studying with friends, but some of them randomly leave the room. You can't rely on them, so you have to learn the material yourself. That's what dropout does—it forces neurons to learn independently.
# 
# 2. **Callbacks**:
   # - Think of them like a coach who watches you surf. If you keep wiping out, the coach stops the session and says, "Let's try again tomorrow." If you're doing great, the coach saves your best run.
# 
# 3. **Validation Split**:
   # - This is like a surprise test. The AI trains on 80% of the data and gets tested on the remaining 20% to make sure it's actually learning, not just memorizing.
# 
# 4. **Early Stopping**:
   # - If the AI isn't improving after 3 rounds, the coach stops training to save time (like quitting a boring game).
# 
# 5. **Model Checkpoint**:
   # - Saves the best version of the AI's brain so you don't lose progress (like saving your game).
    # 

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np  # For number crunching (like a calculator on steroids)

# 1. THE ARCHITECTURE (The "Shape-Shifter")
# -----------------------------------------
# This is like building a robot brain with layers of neurons.
# - Flatten: Unrolls the 28x28 pixel images into a flat list (like spreading out a pancake).
# - Dense: A "thinking layer" where neurons talk to each other (like a group chat).
# - Dropout: Randomly turns off neurons to prevent overfitting (like blindfolding some surfers so they don't copy each other).
# - Softmax: The final layer that picks the best answer (like voting for the best wave).
model = models.Sequential([
    # Step 1: The Scanner (Unroll the image)
    # Input shape (28, 28) = 28x28 pixels (grayscale, so no color channel).
    # In JS: Like `Array.flat()` but for images.
    layers.Flatten(input_shape=(28, 28)),

    # Step 2: The First Brain Layer (512 neurons)
    # ReLU activation: If a neuron's output is negative, it gets turned off (like a surfer who wipes out).
    # In JS: Like `Math.max(0, x)` for every neuron.
    layers.Dense(512, activation='relu'),

    # Step 3: The Anti-Cheating Layer (Dropout 20%)
    # Randomly turns off 20% of neurons during training (so they don't rely on each other).
    # In JS: Like `if (Math.random() < 0.2) neuron.output = 0;`.
    layers.Dropout(0.2),

    # Step 4: The Second Brain Layer (256 neurons)
    # Another layer of thinking, but smaller (like a surfer focusing on fewer waves).
    layers.Dense(256, activation='relu'),

    # Step 5: Another Anti-Cheating Layer (Dropout 20%)
    # More dropout to keep the AI honest.
    layers.Dropout(0.2),

    # Step 6: The Final Decision (10 neurons, one for each digit 0-9)
    # Softmax: Picks the neuron with the highest confidence (like choosing the best wave to ride).
    # In JS: Like `Math.max(...outputs)` but with probabilities.
    layers.Dense(10, activation='softmax')
])

# 2. THE COMPILER (The "Brain Juice")
# -----------------------------------
# This sets up how the AI learns:
# - Optimizer: 'adam' is like a smart surfboard that adjusts to the waves.
# - Loss: 'sparse_categorical_crossentropy' measures how wrong the AI is (like counting wipeouts).
# - Metrics: 'accuracy' tracks how often the AI gets it right (like counting successful rides).
model.compile(
    optimizer='adam',  # The "surfboard tuner" (adjusts weights to ride waves better)
    loss='sparse_categorical_crossentropy',  # How wrong the AI is (like counting wipeouts)
    metrics=['accuracy']  # How often the AI gets it right (like counting successful rides)
)

# 3. THE SMART CALLBACKS (The "Coach")
# -------------------------------------
# Callbacks are like having a coach who watches the AI train and steps in if something's wrong.

# Callback 1: Early Stopping (The "Over It" Coach)
# If the AI's performance doesn't improve for 3 rounds (epochs), the coach stops training.
# In JS: Like `if (last3Scores.every(score => score === currentScore)) stopTraining();`.
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',  # Watch the "validation loss" (how wrong the AI is on the test set)
    patience=3  # Wait 3 rounds before giving up
)

# Callback 2: Model Checkpoint (The "Save the Best" Coach)
# Saves the best version of the AI's brain (so you don't lose progress).
# In JS: Like `localStorage.setItem('bestBrain', JSON.stringify(brain));`.
save_best = callbacks.ModelCheckpoint(
    'best_brain.h5',  # Save the best brain to this file
    save_best_only=True  # Only save if it's better than the last one
)

# 4. THE DATA (The "Training Waves")
# ----------------------------------
# MNIST is a dataset of handwritten digits (0-9), like a library of waves to practice on.
# - x_train: Images of digits (28x28 pixels).
# - y_train: Labels (the correct answers, like "this is a 3").
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data: Scale pixel values from 0-255 to 0-1 (like adjusting wave height for training).
# In JS: Like `x_train.map(pixel => pixel / 255);`.
x_train = x_train / 255.0
x_test = x_test / 255.0

# 5. THE TRAINING (The "Surf Session")
# ------------------------------------
# Now we let the AI train on the data, with the coach watching.
print("Coach is watching... training starts now!")

# Fit the model:
# - x_train, y_train: The training data (waves and correct answers).
# - epochs=50: Train for 50 rounds (like 50 days of surfing).
# - validation_split=0.2: Use 20% of the data as a test set (to check for cheating).
# - callbacks: The coach tools we set up earlier.
history = model.fit(
    x_train,
    y_train,
    epochs=50,
    validation_split=0.2,  # 20% of data is used to test the AI (like a surprise quiz)
    callbacks=[early_stop, save_best]  # The coach tools
)

# 6. THE SUMMARY (The "After-Action Report")
# -----------------------------------------
# If training stops early (e.g., at Epoch 12 instead of 50), it means the coach stepped in.
# Check if 'best_brain.h5' appeared in your folder—that's your winning brain!
print("Training complete! Check if 'best_brain.h5' appeared in your folder. That's your W, dude!")