#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# """
# BORROWING THE RIZZ - EXERCISE 3: USING YOUR OWN DATASET
# Now we're training the model on YOUR OWN IMAGES instead of CIFAR-10.
# Think of this like:
# - You (JS Dev) = Still a noob, but now you're training on **your own surfing footage**.
# - MobileNetV2 = Kelly Slater (still the pro brain).
# - training-data folder = Your shitty GoPro footage of you wiping out at the beach.
# """
# 
# **Organize your `training-data` folder** like this:
# training-data/
   # ├── class1/  # e.g., "wipeout"
   # │   ├── img1.jpg
   # │   ├── img2.jpg
   # │   └── ...
   # └── class2/  # e.g., "shredding"
       # ├── img1.jpg
       # ├── img2.jpg
       # └── ...
        
### **Key Differences from Exercises 1 & 2**
# | Feature          | Exercise 1 & 2 (CIFAR-10) | Exercise 3 (Your Data) |
# |------------------|---------------------------|------------------------|
# | **Dataset**      | CIFAR-10 (60k tiny images) | Your own images (from `training-data` folder) |
# | **Data Loading** | `tf.keras.datasets.cifar10.load_data()` | `ImageDataGenerator` (loads from folders) |
# | **Augmentation** | None (just resizing) | **Heavy augmentation** (rotation, shifts, flips) |
# | **Classes**      | 10 (airplane, car, etc.) | **Your classes** (e.g., "wipeout" vs. "shredding") |
# | **Flexibility**  | Fixed (CIFAR-10 only) | **Fully customizable** (use any images you want) |
# 

# 1. THE SETUP (Import the shit we need)
import tensorflow as tf  # The AI brain framework (like React for AI)
from tensorflow.keras import layers, models  # Tools to build the AI brain (like components in React)
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For loading your own images (like importing shitty GoPro footage)
import numpy as np  # Math stuff (like using lodash for numbers)
import time  # Time tracking (like setTimeout in JS, but for Python)

# 2. DOWNLOAD THE "PRO BRAIN" (MobileNetV2)
# We say include_top=False to chop off its original "head" (output layer)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),  # Shape of the images (224x224 pixels, 3 colors = RGB)
    include_top=False,  # We don't want its original brain, just the body
    weights='imagenet'  # Pre-trained weights (like Kelly's muscle memory)
)

# Freeze the base model (so it doesn't forget Kelly's skills)
# This is like putting Kelly Slater in a straightjacket so he can't move his arms while you try to copy his stance.
base_model.trainable = False

# 3. BUILD YOUR OWN "SHITTY BRAIN" ON TOP
# Now we add our own layers on top of the "pro brain" we borrowed.
# This is like duct-taping a GoPro to Kelly Slater's board so you can record his moves.
model = models.Sequential([
    base_model,  # The pro brain (Kelly Slater)
    layers.GlobalAveragePooling2D(),  # Squish the 3D data into 1D (like flattening a surfboard into a pancake)
    layers.Dense(128, activation='relu'),  # A layer with 128 neurons (like adding 128 tiny surf instructors to your brain)
    layers.Dropout(0.2),  # Randomly turn off 20% of neurons (like randomly forgetting 20% of Kelly's tips)
    layers.Dense(2, activation='softmax')  # Final layer for 2 classes (e.g., "wipeout" vs. "shredding")
])

# 4. COMPILE THE MODEL (Get it ready for training)
# This is like setting up your GoPro to record in 4K before you start filming.
model.compile(
    optimizer='adam',  # The optimizer (like a surf coach who adjusts your stance)
    loss='sparse_categorical_crossentropy',  # The loss function (like a scoreboard for your surfing)
    metrics=['accuracy']  # What we track (like counting how many waves you don't wipe out on)
)

# 5. LOAD YOUR "SHITTY DATASET" (FROM training-data FOLDER)
# This is like importing your shitty GoPro footage instead of watching Kelly's highlight reel.
# We use ImageDataGenerator to load images from folders (like organizing your footage into "wipeout" and "shredding" folders).
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values (0-255 → 0-1) (like adjusting brightness)
    rotation_range=20,  # Randomly rotate images (like your GoPro was mounted sideways)
    width_shift_range=0.2,  # Randomly shift images horizontally (like your GoPro was crooked)
    height_shift_range=0.2,  # Randomly shift images vertically (like your GoPro was too high/low)
    horizontal_flip=True,  # Randomly flip images (like your GoPro was facing the wrong way)
    validation_split=0.2  # Use 20% of data for validation (like holding back some footage for testing)
)

# Load training data (80% of images)
train_generator = train_datagen.flow_from_directory(
    'training-data',  # Path to your training data folder (like "GoPro Footage")
    target_size=(224, 224),  # Resize images to 224x224 (like stretching shitty footage to 4K)
    batch_size=32,  # Number of images per batch (like how many waves you can remember at once)
    class_mode='sparse',  # Use sparse labels (like "0 = wipeout", "1 = shredding")
    subset='training'  # This is the training subset
)

# Load validation data (20% of images)
validation_generator = train_datagen.flow_from_directory(
    'training-data',  # Same folder as training data
    target_size=(224, 224),  # Resize images to 224x224
    batch_size=32,  # Number of images per batch
    class_mode='sparse',  # Use sparse labels
    subset='validation'  # This is the validation subset
)

# Print class indices (so you know what's what)
print("🏄‍♂️ Class indices (what the model thinks your folders mean):")
print(train_generator.class_indices)

# 6. TRAIN THE MODEL (Start copying Kelly's moves)
print("🏄‍♂️ Starting training... (This is where the magic happens, brah)")
start_time = time.time()  # Start the timer (like hitting "record" on your GoPro)

# Train the model (like practicing surfing for 10 epochs)
history = model.fit(
    train_generator,  # Your shitty GoPro footage (training data)
    epochs=10,  # Number of training cycles (like 10 days at surf camp)
    validation_data=validation_generator  # Validation data (like your first real surf session)
)

# Stop the timer and print how long it took
training_time = time.time() - start_time
print(f"🕒 Training took {training_time:.2f} seconds (or {training_time/60:.2f} minutes)")
print("🎉 Training complete! Now go shred some waves, dude!")

# 7. EVALUATE THE MODEL (See if your GoPro footage helped)
# This is like watching the footage and realizing you still look like a drunk kangaroo.
test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
print(f"\n📊 Validation accuracy: {test_acc:.4f} (If this is below 0.5, you're cooked)")

# 8. SAVE THE MODEL (So you don't have to train it again)
# This is like saving your GoPro footage so you can watch it later (and cringe).
model.save('borrowed_rizz_custom_data.h5')
print("💾 Model saved as 'borrowed_rizz_custom_data.h5' (Now you can pretend you're Kelly Slater)")

# ========== EXERCISE SECTION (For the brave souls) ==========
# """
# EXERCISE 4: DATA AUGMENTATION EXPERIMENTS
# The ImageDataGenerator is doing some **gnarly** data augmentation (like adding fake waves to your footage).
# Try tweaking these parameters to see if it improves accuracy:
# - rotation_range=40 (more rotation = more fake waves)
# - zoom_range=0.2 (zoom in/out = more fake distances)
# - brightness_range=[0.5, 1.5] (adjust brightness = more fake lighting)
# """
# 
# """
# EXERCISE 5: TRY A DIFFERENT MODEL
# MobileNetV2 is cool, but what if we want to borrow from a different pro?
# Try swapping it with ResNet50 or EfficientNet.
# This is like trying to copy John John Florence instead of Kelly Slater.
# 
# # Example with ResNet50:
# base_model = tf.keras.applications.ResNet50(
    # input_shape=(224, 224, 3),
    # include_top=False,
    # weights='imagenet'
# )
# """
# 
# """
# EXERCISE 6: FINE-TUNING (UNFREEZE LAYERS)
# Right now, we froze the entire base_model. But what if we want to fine-tune the last few layers?
# Uncomment the code below to unfreeze the last 20 layers of MobileNetV2 and train them too.
# This is like letting Kelly Slater adjust your stance a little bit.
# 
# base_model.trainable = True
# for layer in base_model.layers[:-20]:
    # layer.trainable = False
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate
              # loss='sparse_categorical_crossentropy',
              # metrics=['accuracy'])
# history = model.fit(train_generator, epochs=5, validation_data=validation_generator)
# """