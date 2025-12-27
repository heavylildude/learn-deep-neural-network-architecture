#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# DNN W2: The _Oopsie_ Rewind - Python Edition
# ============================================
# This script is like a **JS setTimeout()** but for AI training.
# It simulates how a neural network "learns" by adjusting its guess
# until it hits the **Perfect Answer (42.0)**.
# 
# Think of it like this:
# - In JS, you'd do `setTimeout(() => { ... }, 1000)`
# - In Python, we use `time.sleep(1)` to pause execution.
# - The "learning" part is like a `while` loop in JS that keeps running
  # until the AI gets it right.


# 1. THE SETUP (Like declaring variables in JS)
import time  # JS equivalent: `setTimeout()` or `setInterval()`
import numpy as np  # JS equivalent: `Math` but on steroids (for AI math)

# The "Perfect" Answer (like a hardcoded value in JS)
# In JS: `const targetNumber = 42.0;`
target_number = 42.0

# Where the AI starts (like `let currentGuess = 0;` in JS)
current_guess = 0.0

# How fast the AI learns (like `let learningRate = 0.1;` in JS)
# Too high = AI overshoots (like a drunk guy trying to walk straight)
# Too low = AI takes forever (like a snail on molasses)
learning_rate = 0.1

# 2. THE LEARNING LOOP (Like a `while` loop in JS)
# In JS: `while (currentGuess !== targetNumber) { ... }`
while abs(current_guess - target_number) > 0.001:  # "Close enough" check
    # 2.1. THE "OOPSIE" (How wrong is the AI?)
    # In JS: `const error = targetNumber - currentGuess;`
    error = target_number - current_guess

    # 2.2. THE "REWIND" (Adjust the guess)
    # In JS: `currentGuess += learningRate * error;`
    # This is like the AI saying:
    # "Oh shit, I was off by X. Let me nudge my guess a little closer."
    current_guess += learning_rate * error

    # 2.3. THE "PAUSE" (Like `setTimeout()` in JS)
    # In JS: `setTimeout(() => { ... }, 1000);`
    time.sleep(1)  # Wait 1 second (so you can see the AI "thinking")

    # 2.4. THE "DEBUG LOG" (Like `console.log()` in JS)
    # In JS: `console.log(`Current guess: ${currentGuess}`);`
    print(f"Current guess: {current_guess} (Error: {error})")

# 3. THE "EUREKA!" MOMENT (Like `console.log("Got it!");` in JS)
print(f"🎉 AI found the answer! It's {current_guess} (Target was {target_number})")