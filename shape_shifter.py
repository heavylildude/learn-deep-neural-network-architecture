# =============================================
# SHAPE-SHIFTER CNN - THE AI BRAIN THAT SEES
# =============================================
# Imagine this shit like a **super-smart pancake stack** that can recognize handwritten numbers.
# It's like a **digital detective** that looks at squiggly lines and goes "Yo, that's a 4, brah!"

import tensorflow as tf
from tensorflow.keras import layers, models

# =============================================
# 1. THE ARCHITECTURE (The Shape-Shifter's Brain)
# =============================================
# This is like building a **robot brain** with different layers.
# Each layer does a **specific job**, like a **team of detectives** working together.

model = models.Sequential([
    # -----------------------------------------
    # Step 1: The Scanner (Conv2D)
    # -----------------------------------------
    # This is like the **first detective** who looks at the image and finds **simple shapes** (lines, curves, etc.).
    # - `32`: Number of detectives (filters) looking for different shapes.
    # - `(3, 3)`: Size of the detective's magnifying glass (how big of an area they look at).
    # - `activation='relu'`: If the detective finds something, they **shout it out** (ReLU = "If it's important, say it!").
    # - `input_shape=(28, 28, 1)`: The image is 28x28 pixels (black & white, so 1 channel).
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    # -----------------------------------------
    # Step 2: The TL;DR (MaxPooling2D)
    # -----------------------------------------
    # This is like the **boss detective** who **summarizes** what the first detectives found.
    # - `(2, 2)`: The boss looks at **2x2 blocks** and picks the **most important clue** (max value).
    # This makes the image **smaller** (like squishing a pancake to make it thinner).
    layers.MaxPooling2D((2, 2)),

    # -----------------------------------------
    # Step 3: Another Scanner (Conv2D)
    # -----------------------------------------
    # Now, a **second team of detectives** looks for **more complex shapes** (like circles, loops, etc.).
    # - `64`: More detectives this time (because complex shapes need more eyes).
    # - `(3, 3)`: Same magnifying glass size.
    # - `activation='relu'`: Same shouting rule ("If it's important, say it!").
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Another TL;DR (MaxPooling2D)
    # The boss detective **summarizes again** to make the image even smaller.
    layers.MaxPooling2D((2, 2)),

    # -----------------------------------------
    # Step 4: The Decision Maker (Flatten + Dense)
    # -----------------------------------------
    # Now, the **final boss** (a **super-smart AI judge**) looks at all the clues and makes the **final decision**.

    # Flatten: Turns the **2D image clues** into a **long list of numbers** (like unrolling a pancake into a noodle).
    layers.Flatten(),

    # Dense (64): The **first brain layer** of the judge.
    # - `64`: Number of **brain cells** (neurons) thinking about the clues.
    # - `activation='relu'`: Same shouting rule ("If it's important, say it!").
    layers.Dense(64, activation='relu'),

    # Dense (10): The **final decision layer**.
    # - `10`: Because there are **10 possible numbers** (0-9).
    # - `activation='softmax'`: The judge **picks the most likely answer** (like "This is 80% a 4, 10% a 9, etc.").
    layers.Dense(10, activation='softmax')
])

# =============================================
# 2. THE COMPILATION (Teaching the AI)
# =============================================
# Now we **teach the AI** how to learn. It's like giving it a **study guide** and a **grading system**.

# - `optimizer='adam'`: The **study method** (Adam = "Smart way to learn without getting stuck").
# - `loss='sparse_categorical_crossentropy'`: The **grading system** (How wrong the AI is when it guesses).
# - `metrics=['accuracy']`: How we **measure success** (Like a test score: "You got 95% right, brah!").
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# =============================================
# 3. SUMMARY (Let's See the Brain)
# =============================================
# This prints out a **map of the AI brain** so we can see how many **brain cells (params)** it has.
# Think of it like counting how many **neurons** are in a **digital pancake stack**.
model.summary()
print("\nBrah, look at that 'Param' count. That's a lot of Rizz in one model.")

# =============================================
# 4. EXERCISE (Try It Yourself!)
# =============================================
# Go back to your **Week 3 MNIST code** and replace the simple model with this **CNN**.
# Watch the **accuracy fly higher than a kickflip on a vert ramp**!
# (This is like upgrading from a **bicycle** to a **rocket ship** for number recognition.)