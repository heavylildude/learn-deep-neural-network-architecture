#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BORROWING THE RIZZ - EXERCISE 2: SWAPPING THE PRO BRAIN
Now we're swapping MobileNetV2 (Kelly Slater) for ResNet50 (John John Florence).
Think of this like:
- You (JS Dev) = Still a noob surfer, but now you're trying to copy **John John Florence** instead of Kelly Slater.
- ResNet50 = John John Florence (another pro, but with a different style).
- This script = You watching John John's YouTube tutorials and trying to copy his **buttery smooth turns**.
"""

# 1. THE SETUP (Import the shit we need)
import tensorflow as tf  # The AI brain framework (like React for AI)
from tensorflow.keras import layers, models  # Tools to build the AI brain (like components in React)
import numpy as np  # Math stuff (like using lodash for numbers)
import time  # Time tracking (like setTimeout in JS, but for Python)

# 2. DOWNLOAD THE "PRO BRAIN" (ResNet50)
# We say include_top=False to chop off its original "head" (output layer)
# This is like removing John John Florence's surfboard fins and replacing them with your own shitty fins.
base_model = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3),  # Shape of the images (224x224 pixels, 3 colors = RGB)
    include_top=False,  # We don't want its original brain, just the body
    weights='imagenet'  # Pre-trained weights (like John John's muscle memory)
)

# Freeze the base model (so it doesn't forget John John's skills)
# This is like putting John John in a straightjacket so he can't move his arms while you try to copy his stance.
base_model.trainable = False

# 3. BUILD YOUR OWN "SHITTY BRAIN" ON TOP
# Now we add our own layers on top of the "pro brain" we borrowed.
# This is like duct-taping a GoPro to John John's board so you can record his moves.
model = models.Sequential([
    base_model,  # The pro brain (John John Florence)
    layers.GlobalAveragePooling2D(),  # Squish the 3D data into 1D (like flattening a surfboard into a pancake)
    layers.Dense(128, activation='relu'),  # A layer with 128 neurons (like adding 128 tiny surf instructors to your brain)
    layers.Dropout(0.2),  # Randomly turn off 20% of neurons (like randomly forgetting 20% of John John's tips)
    layers.Dense(10, activation='softmax')  # Final layer for 10 classes (like choosing between 10 different surf tricks)
])

# 4. COMPILE THE MODEL (Get it ready for training)
# This is like setting up your GoPro to record in 4K before you start filming.
model.compile(
    optimizer='adam',  # The optimizer (like a surf coach who adjusts your stance)
    loss='sparse_categorical_crossentropy',  # The loss function (like a scoreboard for your surfing)
    metrics=['accuracy']  # What we track (like counting how many waves you don't wipe out on)
)

# 5. LOAD YOUR "SHITTY DATASET" (CIFAR-10)
# This is like downloading a bunch of shitty surfing videos from YouTube.
# CIFAR-10 has 60,000 tiny images (32x32 pixels) of 10 classes (airplane, car, bird, etc.)
# We're gonna resize them to 224x224 because ResNet50 is a diva and only works with big images.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Resize the images (like stretching a tiny surfing thumbnail into a 4K video)
x_train = tf.image.resize(x_train, (224, 224))  # Stretch training images
x_test = tf.image.resize(x_test, (224, 224))  # Stretch test images

# Normalize pixel values (0-255 → 0-1)
# This is like adjusting the brightness of your surfing videos so they're easier to watch.
x_train = x_train / 255.0
x_test = x_test / 255.0

# 6. TRAIN THE MODEL (Start copying John John's moves)
print("🏄‍♂️ Starting training... (This is where the magic happens, brah)")
start_time = time.time()  # Start the timer (like hitting "record" on your GoPro)

# Train the model (like practicing surfing for 5 epochs)
history = model.fit(
    x_train, y_train,  # Your shitty dataset (training data)
    epochs=5,  # Number of training cycles (like 5 days at surf camp)
    validation_data=(x_test, y_test)  # Test data (like your first real surf session)
)

# Stop the timer and print how long it took
training_time = time.time() - start_time
print(f"🕒 Training took {training_time:.2f} seconds (or {training_time/60:.2f} minutes)")
print("🎉 Training complete! Now go shred some waves, dude!")

# 7. EVALUATE THE MODEL (See how shitty your surfing is)
# This is like watching the GoPro footage and realizing you look like a drunk kangaroo.
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n📊 Test accuracy: {test_acc:.4f} (If this is below 0.5, you're cooked)")

# 8. SAVE THE MODEL (So you don't have to train it again)
# This is like saving your GoPro footage so you can watch it later (and cringe).
model.save('borrowed_rizz_resnet50.h5')
print("💾 Model saved as 'borrowed_rizz_resnet50.h5' (Now you can pretend you're John John Florence)")

# ========== EXERCISE SECTION (For the brave souls) ==========
# """
# SUB-EXERCISE 1: UNFREEZE SOME LAYERS
# Right now, we froze the entire base_model. But what if we want to fine-tune the last few layers?
# Uncomment the code below to unfreeze the last 20 layers of ResNet50 and train them too.
# This is like letting John John Florence adjust your stance a little bit.
# 
# base_model.trainable = True
# for layer in base_model.layers[:-20]:
    # layer.trainable = False
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate
              # loss='sparse_categorical_crossentropy',
              # metrics=['accuracy'])
# history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
# """
# 
# """
# SUB-EXERCISE 2: TRY ANOTHER MODEL
# ResNet50 is cool, but what if we want to borrow from a different pro?
# Try swapping ResNet50 with another model like EfficientNet or VGG16.
# This is like trying to copy Italo Ferreira instead of John John Florence.
# 
# # Example with EfficientNet:
# base_model = tf.keras.applications.EfficientNetB0(
    # input_shape=(224, 224, 3),
    # include_top=False,
    # weights='imagenet'
# )
# """