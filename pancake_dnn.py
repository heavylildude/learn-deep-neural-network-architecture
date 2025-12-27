#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# PANCAKE DNN - The AI that learns to read handwritten numbers like a boss.
# Think of this like teaching a robot to recognize your shitty handwriting.
# """

# 1. THE SETUP (Gathering the Ingredients)
import tensorflow as tf
from tensorflow.keras import layers, models

# JS ANALOGY:
# Imagine you're importing "react" and "react-dom" to build a UI.
# Here, we're importing "tensorflow" to build an AI brain.

# 2. THE DATA (Handwritten Numbers 0-9)
# MNIST is like a giant photo album of handwritten numbers (0-9)
# Each photo is 28x28 pixels (black and white, no colors)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# JS ANALOGY:
# This is like fetching an array of images from an API.
# x_train = photos of numbers (the "pixels")
# y_train = the actual numbers (the "labels")
# x_test/y_test = the "final exam" photos/labels

# Normalize: Make pixels 0-1 instead of 0-255 (Better vibes)
# Why? Because AI brains work better with small numbers (like how you prefer your pizza not burnt to a crisp)
x_train, x_test = x_train / 255.0, x_test / 255.0

# JS ANALOGY:
# This is like converting RGB values (0-255) to opacity (0-1) in CSS.
# It makes the math easier for the AI.

# 3. THE ARCHITECTURE (Stacking the Pancakes)
# We're building a "Sequential" model, which means layers stacked like pancakes.
# Each layer is like a worker in a factory line, passing info to the next.
model = models.Sequential([
    # Layer 1: Flatten (The Pancake Press)
    # Flattens the 28x28 photo into a 1D line (784 pixels long)
    # Imagine taking a photo and stretching it into a single line of pixels.
    layers.Flatten(input_shape=(28, 28)),

    # JS ANALOGY:
    # This is like using .flat() on a nested array in JS.
    # [[1,2], [3,4]] → [1,2,3,4]

    # Layer 2: Dense (The "Smart" Layer)
    # 128 neurons = 128 tiny workers looking for patterns in the pixels.
    # "relu" = If a worker sees something useful, they shout it out. If not, they stay quiet.
    layers.Dense(128, activation='relu'),

    # JS ANALOGY:
    # This is like a .map() function that processes each pixel.
    # "relu" is like a filter that removes negative values (noise).

    # Layer 3: Dense (The "Smarter" Layer)
    # 64 neurons = Fewer workers, but they're more specialized.
    # They take the patterns from the first layer and refine them.
    layers.Dense(64, activation='relu'),

    # JS ANALOGY:
    # This is like chaining another .map() to refine the data further.

    # Layer 4: Output Layer (The Final Answer)
    # 10 neurons = One for each digit (0-9).
    # "softmax" = The worker with the loudest shout wins (highest probability).
    layers.Dense(10, activation='softmax')
])

# JS ANALOGY:
# This is like a .reduce() function that picks the most confident answer.
# [0.1, 0.7, 0.2] → "The answer is 1 (70% confidence)"

# 4. THE TRAINING (The Grind)
# "compile" = Setting up the training rules.
# - optimizer='adam': The "coach" that adjusts the workers' shouts to improve accuracy.
# - loss='sparse_categorical_crossentropy': The "scoreboard" that tracks how wrong the AI is.
# - metrics=['accuracy']: How we measure success (like a report card).
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# JS ANALOGY:
# This is like setting up a training loop with:
# - A "coach" (optimizer) to improve performance.
# - A "scoreboard" (loss) to track mistakes.
# - A "report card" (metrics) to show progress.

print("Starting the grind... watch those layers learn!")
# "fit" = Train the model on the data for 3 "epochs" (full passes through the data).
# JS ANALOGY:
# This is like running a for-loop 3 times to train the AI.
model.fit(x_train, y_train, epochs=3)

# 5. THE TEST (The Final Exam)
# "evaluate" = Test the model on unseen data (the "final exam").
# Returns the loss (how wrong it is) and accuracy (how right it is).
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nFinal Brain Accuracy: {test_acc*100:.2f}% - Absolute Legend!")

# JS ANALOGY:
# This is like running unit tests to check if your code works.
# If test_acc > 95%, your AI is a fucking genius.