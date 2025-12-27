# =============================================
# 1. THE SETUP (Gathering Our Tools)
# =============================================
# Think of these like your JS imports, but for Python AI shit.
import tensorflow as tf  # The AI brain (like TensorFlow.js but in Python)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  # Magic image tools
import matplotlib.pyplot as plt  # Like a JS canvas, but for showing images
import numpy as np  # Math magic (like NumPy in JS, but way more powerful)

# =============================================
# 2. THE CLONE MACHINE (Our Magic Spinner)
# =============================================
# This is like a JS function that takes an image and spins it into 10 variations.
# Imagine a pizza spinning in the air—this is what we're doing to your image!
datagen = ImageDataGenerator(
    rotation_range=40,      # Spin it like a pizza! (0-40 degrees)
    width_shift_range=0.2,  # Slide it left/right (like a JS `transform: translateX()`)
    height_shift_range=0.2, # Slide it up/down (like a JS `transform: translateY()`)
    shear_range=0.2,        # Skew it like a parallelogram (like `transform: skew()` in CSS)
    zoom_range=0.2,         # Zoom in/out (like `transform: scale()` in CSS)
    horizontal_flip=True,   # Mirror it (like `transform: scaleX(-1)` in CSS)
    fill_mode='nearest'     # Fill gaps with nearby pixels (like `background-repeat: no-repeat` in CSS)
)

# =============================================
# 3. LOAD THE ORIGINAL IMAGE (Put Your Pizza Here)
# =============================================
# This is like `document.getElementById('image')` in JS, but for Python.
try:
    # Load the image (like `new Image()` in JS)
    img = load_img('test.jpg')  # Replace 'test.jpg' with your image filename

    # Convert the image to numbers (like `ctx.getImageData()` in JS canvas)
    x = img_to_array(img)  # Turns the image into a 3D array (width, height, RGB channels)

    # Reshape it so the AI brain can understand it (like adding a batch dimension in TensorFlow.js)
    x = x.reshape((1,) + x.shape)  # Adds a "batch" dimension (1 image at a time)
except:
    # If the user forgets to add 'test.jpg', we use random noise (like a fallback in JS)
    print("Brah, you need a 'test.jpg' in this folder! Using random noise instead...")
    x = np.random.rand(1, 150, 150, 3)  # Random noise (1 image, 150x150 pixels, 3 RGB channels)

# =============================================
# 4. GENERATE THE CLONES (Spin the Pizza!)
# =============================================
# This is like a `for` loop in JS, but for generating image variations.
print("Creating 10 clones of your image... watch the rizz happen.")

# Set up a big canvas to show all the clones (like `document.createElement('canvas')` in JS)
plt.figure(figsize=(10, 10))  # 10x10 inches canvas

# Loop through the clone machine (like `for (let i = 0; i < 10; i++)` in JS)
i = 0
for batch in datagen.flow(x, batch_size=1):  # `flow()` is like a JS generator
    # Add the cloned image to our canvas (like `ctx.drawImage()` in JS)
    plt.subplot(3, 4, i + 1)  # Arrange clones in a 3x4 grid (like CSS Grid)
    plt.imshow(batch[0].astype('uint8'))  # Show the image (convert numbers to pixels)

    i += 1
    if i % 10 == 0: break  # Stop after 10 clones (like `break` in JS)

# Show the canvas (like `canvas.style.display = 'block'` in JS)
plt.show()