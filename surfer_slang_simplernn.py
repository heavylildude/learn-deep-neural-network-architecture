#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SURFER SLANG GENERATOR (The "Digital Shaka Brain - SimpleRNN Edition")
-----------------------------------------------------------------------
This script teaches an AI to talk like a **gnarly** surfer, but this time
we're using a **SimpleRNN**—the "kook" version of an LSTM.

Think of SimpleRNN like a **boogie board**:
- 🏄‍♂️ **Easy to ride** (simple architecture)
- 🌊 **Forgets shit fast** (no long-term memory)
- 🔥 **Good for short waves** (short sequences)

LSTM is like a **performance shortboard**:
- 🏄‍♂️ **Harder to ride** (complex architecture)
- 🌊 **Remembers everything** (long-term memory)
- 🔥 **Shreds big waves** (long sequences)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding  # CHANGED: LSTM → SimpleRNN
import random
import time  # For dramatic effect (like JS's setTimeout)

# =============================================
# 1. THE DATA (Our "Surf Dictionary")
# =============================================
# Same as before—this is our "training data" (like a JS array of words).
text = "stoked gnarly rad wicked shred send it kook wipeout"
chars = sorted(list(set(text)))  # Like JS's new Set() - removes duplicates
char_to_int = dict((c, i) for i, c in enumerate(chars))  # JS object: { char: index }
int_to_char = dict((i, c) for i, c in enumerate(chars))  # Reverse lookup: { index: char }

# =============================================
# 2. PREPARE THE TRAINING DATA (The "Surf Lessons")
# =============================================
# We split the text into sequences (like JS's .slice()).
maxlen = 5  # Like JS's .slice(0, 5)
sentences = []  # JS array to store sequences
next_chars = []  # JS array to store the "next char" (like labels)
for i in range(0, len(text) - maxlen):
    sentences.append(text[i:i + maxlen])  # JS: text.slice(i, i + maxlen)
    next_chars.append(text[i + maxlen])  # JS: text[i + maxlen]

# =============================================
# 3. TURN CHARS INTO NUMBERS (The "Surf Code")
# =============================================
# Convert each char to its index (like JS's .map()).
x = np.zeros((len(sentences), maxlen), dtype=np.int32)  # Like JS's Array.fill(0)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool_)  # One-hot encoding (like JS's Array.fill(false))
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t] = char_to_int[char]  # JS: char_to_int[char]
    y[i, char_to_int[next_chars[i]]] = 1  # JS: y[i][char_to_int[next_chars[i]]] = true

# =============================================
# 4. BUILD THE AI MODEL (The "Digital Boogie Board")
# =============================================
# This is like a **JS function** that takes input (previous chars)
# and predicts the next char—but **SimpleRNN is simpler** (and dumber).
model = Sequential([
    # 1. Embedding Layer: Like a JS Map() that turns numbers into "vibes"
    Embedding(len(chars), 8, input_length=maxlen),

    # 2. SimpleRNN Layer: The "kook" version of LSTM (like a boogie board vs. a shortboard)
    # - No fancy gates (forget/input/output) → just remembers the last few chars.
    # - Less powerful, but **easier to train** (like learning to surf on a boogie board).
    SimpleRNN(32),  # CHANGED: LSTM(32) → SimpleRNN(32)

    # 3. Dense Layer: The "output" (like a JS switch statement that picks the next char)
    Dense(len(chars), activation='softmax')
])

# Compile the model (like JS's .then() after a Promise)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# =============================================
# 5. TRAIN THE MODEL (The "Surf Session")
# =============================================
# This is like a **JS loop** that keeps adjusting the boogie board
# until it "sorta shreds" the waves (but not as well as an LSTM).
print("🏄‍♂️  Paddling out... (Training the AI - SimpleRNN Edition)")
model.fit(x, y, batch_size=1, epochs=100)  # JS: for (let i = 0; i < 100; i++)

# =============================================
# 6. GENERATE NEW SLANG (The "Boogie Board Mode")
# =============================================
def generate_slang(seed=None, length=20, temperature=0.5):
    """
    Generate **gnarly** surf slang (but SimpleRNN might be a **kook** at it)!

    Args:
        seed (str): Starting text (like a JS prompt)
        length (int): How many chars to generate (like JS's .length)
        temperature (float): How "wild" the AI gets (0.1 = safe, 1.0 = cooked)

    Returns:
        str: A **rad** (or **bogus**) surf phrase
    """
    if seed is None:
        seed = random.choice(sentences)  # Pick a random starting point (like JS's Math.random())
    generated = seed

    print(f"🌊  Starting with: '{seed}'")
    print("🏄‍♂️  Trying to shred the gnar (SimpleRNN style)...")

    for i in range(length):
        # Convert the current text to numbers (like JS's .map())
        x_pred = np.zeros((1, maxlen))
        for t, char in enumerate(generated[-maxlen:]):
            x_pred[0, t] = char_to_int[char]

        # Predict the next char (like JS's .then() in a Promise)
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)  # Pick a char based on "temperature"
        next_char = int_to_char[next_index]

        generated += next_char  # Add the new char (like JS's +=)

        # Print progress (like JS's console.log)
        print(f"🔥  Generated so far: '{generated}'", end='\r')
        time.sleep(0.1)  # Dramatic pause (like JS's setTimeout)

    print(f"\n🤙  Final shred: '{generated}'")
    return generated

def sample(preds, temperature=1.0):
    """
    Pick the next char based on "temperature" (like JS's Math.random() but smarter).

    Args:
        preds (np.array): Predictions from the AI (like a JS array of probabilities)
        temperature (float): How "wild" the AI gets (0.1 = safe, 1.0 = cooked)

    Returns:
        int: The index of the next char (like JS's array index)
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# =============================================
# 7. LET'S SHRED! (The "Main Event")
# =============================================
if __name__ == "__main__":
    print("""
    🌊🏄‍♂️  WELCOME TO THE DIGITAL BOOGIE BRAIN! 🏄‍♂️🌊
    -------------------------------------------------
    This AI is **cooked** with a **SimpleRNN**—it might be a **kook**,
    but it's **easier to train** than an LSTM!

    Try it out and see if it **shreds** or **wipes out**!
    """)

    while True:
        print("\n🔥  Options:")
        print("1. Generate random slang (press Enter)")
        print("2. Start with a custom seed (type your seed)")
        print("3. Exit (type 'exit')")

        user_input = input("\n🤙  What's your move? ")

        if user_input.lower() == 'exit':
            print("🤙  Catch you on the flip side, brah!")
            break
        elif user_input.strip() == "":
            generate_slang()
        else:
            generate_slang(seed=user_input, temperature=0.7)

    print("🤙  Catch you on the flip side, brah!")