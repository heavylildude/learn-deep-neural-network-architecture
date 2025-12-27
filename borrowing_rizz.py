#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BORROWING THE RIZZ - A Python script that teaches your AI brain to "borrow" intelligence
from a pro-level model (MobileNetV2) and fine-tune it for your own shitty dataset.

Think of this like:
- You (JS Dev) = A noob surfer who just learned how to stand on a board
- MobileNetV2 = Kelly Slater (the GOAT)
- This script = You watching Kelly's YouTube tutorials and trying to copy his moves

TL;DR: We're stealing Kelly Slater's surfing skills to look rad at the beach.
"""

# 1. THE SETUP (Import the shit we need)
import tensorflow as tf  # The AI brain framework (like React for AI)
from tensorflow.keras import layers, models  # Tools to build the AI brain (like components in React)
import numpy as np  # Math stuff (like using lodash for numbers)
import time  # Time tracking (like setTimeout in JS, but for Python)

# 2. DOWNLOAD THE "PRO BRAIN" (MobileNetV2)
# We say include_top=False to chop off its original "head" (output layer)
# This is like removing Kelly Slater's surfboard fins and replacing them with your own shitty fins
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),  # Shape of the images (224x224 pixels, 3 colors = RGB)
    include_top=False,  # We don't want its original brain, just the body
    weights='imagenet'  # Pre-trained weights (like Kelly's muscle memory)
)

# Freeze the base model (so it doesn't forget Kelly's skills)
# This is like putting Kelly Slater in a straightjacket so he can't move his arms
# while you try to copy his stance. The base model's weights won't change during training.
base_model.trainable = False

# 3. BUILD YOUR OWN "SHITTY BRAIN" ON TOP
# Now we add our own layers on top of the "pro brain" we borrowed.
# This is like duct-taping a GoPro to Kelly Slater's board so you can record his moves.
model = models.Sequential([
    base_model,  # The pro brain (Kelly Slater)
    layers.GlobalAveragePooling2D(),  # Squish the 3D data into 1D (like flattening a surfboard into a pancake)
    layers.Dense(128, activation='relu'),  # A layer with 128 neurons (like adding 128 tiny surf instructors to your brain)
    layers.Dropout(0.2),  # Randomly turn off 20% of neurons (like randomly forgetting 20% of Kelly's tips)
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
# We're gonna resize them to 224x224 because MobileNetV2 is a diva and only works with big images.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Resize the images (like stretching a tiny surfing thumbnail into a 4K video)
x_train = tf.image.resize(x_train, (224, 224))  # Stretch training images
x_test = tf.image.resize(x_test, (224, 224))  # Stretch test images

# Normalize pixel values (0-255 → 0-1)
# This is like adjusting the brightness of your surfing videos so they're easier to watch.
x_train = x_train / 255.0
x_test = x_test / 255.0

# 6. TRAIN THE MODEL (Start copying Kelly's moves)
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
model.save('borrowed_rizz_model.h5')
print("💾 Model saved as 'borrowed_rizz_model.h5' (Now you can pretend you're Kelly Slater)")

# ========== EXERCISE SECTION (For the brave souls) ==========
"""
EXERCISE 1: UNFREEZE SOME LAYERS
Right now, we froze the entire base_model. But what if we want to fine-tune the last few layers?
Uncomment the code below to unfreeze the last 20 layers of MobileNetV2 and train them too.
This is like letting Kelly Slater adjust your stance a little bit.

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
"""

"""
EXERCISE 2: TRY A DIFFERENT MODEL
MobileNetV2 is cool, but what if we want to borrow from a different pro?
Try swapping MobileNetV2 with another model like ResNet50 or EfficientNet.
This is like trying to copy John John Florence instead of Kelly Slater.

# Example with ResNet50:
base_model = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
"""

"""
EXERCISE 3: USE YOUR OWN DATASET
CIFAR-10 is shit. What if you want to train on your own surfing photos?
Use ImageDataGenerator to load your own images from a folder.
This is like recording your own surfing videos instead of watching Kelly's.

# Example:
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'path/to/your/training_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)
model.fit(train_generator, epochs=5)
"""



# ### **Key Differences Between MobileNetV2 and ResNet50**
# | Feature          | MobileNetV2 (Kelly Slater) | ResNet50 (John John Florence) |
# |------------------|---------------------------|-------------------------------|
# | **Size**         | Lightweight (~14MB)       | Bigger (~100MB)               |
# | **Speed**        | Fast (good for mobile)    | Slower (but more accurate)    |
# | **Style**        | Aggressive, quick turns   | Smooth, buttery turns         |
# | **Best For**     | Edge devices, real-time   | High accuracy, big datasets   |
# 
# ---
# 
# ### **Why This Works**
# - **ResNet50** is **deeper** than MobileNetV2, meaning it has **more layers** (like John John has more experience than Kelly Slater).
# - It uses **skip connections** (like a surfboard with extra fins for stability), which helps it **learn better** without getting "lost" in the data.
# - It’s **slower** but **more accurate**—like how John John takes his time to carve the perfect wave.
# 
# ---
# 
# ### **What’s Next?**
# 1. **Run the script** and see if your **test accuracy** improves (it should, because ResNet50 is a **beast**).
# 2. **Try Exercise 1** (unfreezing layers) to see if fine-tuning helps even more.
# 3. **Experiment with other models** (EfficientNet, VGG16) to find the **best pro brain** for your shitty dataset.
# 

