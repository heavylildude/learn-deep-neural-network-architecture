#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# """
# FINAL DEMO: The "Big Brain Flex" 🧠🔥
# -------------------------------------
# This script is your AI's "final exam". It loads your trained model
# and lets you test it on NEW images—just like a real-world demo!
# 
# Think of this like:
# - JS: `import fs from 'fs'` → Python: `import tensorflow as tf`
# - JS: `const model = await tf.loadLayersModel('model.json')` → Python: `tf.keras.models.load_model('best_brain.h5')`
# """

import tensorflow as tf  # The "AI Brain Library" (like TensorFlow.js in Python)
import numpy as np       # The "Math Magic Library" (like NumJS in Python)
from PIL import Image    # The "Image Helper" (like `new Image()` in JS)

# =============================================
# 1. LOAD YOUR BEST BRAIN 🧠 (The "Model Checkpoint")
# =============================================
# In JS: You'd do `await tf.loadLayersModel('model.json')`
# In Python: We use `tf.keras.models.load_model('best_brain.h5')`
#
# Why `.h5`? It's like a `.zip` file for your AI brain—holds all the weights!
# If it fails, it's like trying to load a `.js` file that doesn't exist.
try:
    model = tf.keras.models.load_model('best_brain.h5')
    print("🔥 AI IS LOADED AND READY TO SLAY! (Like a fully charged PlayStation!)")
except Exception as e:
    print(f"🤬 D'oh! No model found. Use your Week 8 weights, brah! Error: {e}")
    # JS analogy: `console.error("Model not found!");`

# =============================================
# 2. THE RIZZ WRAPPER 🎁 (The "Prediction Function")
# =============================================
# This is like a JS function:
# `async function predictWithRizz(imagePath) { ... }`
#
# It takes an image path (like `'pizza.jpg'`) and returns the AI's guess.
def predict_with_rizz(image_path):
    """
    AI's "Guessing Machine" 🤖
    -------------------------
    Steps:
    1. Open the image (like `new Image()` in JS)
    2. Resize it (like `image.resize(160, 160)`)
    3. Convert to numbers (like `tf.browser.fromPixels(image)`)
    4. Feed to AI (like `model.predict(tensor)`)
    5. Return the vibe (like `console.log("BUSSIN!");`)
    """

    # Step 1: Open the image (like `new Image()` in JS)
    # - `Image.open()` is like `new Image()` in JS
    # - `.resize()` is like `image.width = 160; image.height = 160;`
    img = Image.open(image_path).resize((160, 160))  # Adjust to YOUR model's input size!

    # Step 2: Convert to numbers (like `tf.browser.fromPixels(image)` in JS)
    # - `np.array(img)` turns the image into a "number grid" (like a 3D array in JS)
    # - `/ 255.0` normalizes it (like `tensor.div(tf.scalar(255))` in JS)
    img_array = np.array(img) / 255.0

    # Step 3: Add a "batch dimension" (like `tf.expandDims(tensor, 0)` in JS)
    # - AI expects `[batch_size, height, width, channels]`
    # - Here, batch_size = 1 (just one image)
    img_array = np.expand_dims(img_array, axis=0)

    # Step 4: The AI makes a guess (like `model.predict(tensor)` in JS)
    # - `model.predict()` is like `await model.predict(tensor)` in JS
    # - Returns a "confidence score" (like `[0.2, 0.8]` for 2 classes)
    prediction = model.predict(img_array)

    # Step 5: Interpret the vibe (like `if (score > 0.5) { ... }` in JS)
    # - If your model has 2 classes (e.g., "Basic" vs. "Bussin"), it returns `[score_class0, score_class1]`
    # - Here, we assume `prediction[0][0]` is the score for "Basic" (adjust if needed!)
    score = prediction[0][0]

    # Print the AI's "vibe check" (like `console.log()` in JS)
    if score > 0.5:
        print(f"🍕 AI VIBE: BUSSIN! (Confidence: {score*100:.1f}%) 🔥")
    else:
        print(f"🤮 AI VIBE: BASIC... (Confidence: {(1-score)*100:.1f}%) 😬")

# =============================================
# 3. THE LIVE DEMO 🎤 (The "Test Drive")
# =============================================
# Uncomment this line to test your AI on a NEW image!


# predict_with_rizz('secret_test_image.jpg')

