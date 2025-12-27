#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# """
# TRANSFORMERS.PY (The "Vibe-Checker & Storyteller")
# --------------------------------------------------
# This script is like borrowing a **pro surfer's brain** to:
# 1. Check if a comment is **gnarly** (positive) or **bogus** (negative).
# 2. Generate a **rad** surf story using AI.
# 
# Think of it like this:
# - The "Vibe-Checker" is like a **lifeguard** who judges if your vibe is **stoked** or **sucks**.
# - The "Storyteller" is like a **shaka master** who spins a **tubular** tale based on a prompt.
# """

# YOU NEED: pip install transformers torch
# (This is like downloading the **AI's surfboard**—it needs the right gear to shred.)
import time
from transformers import pipeline

# =============================================
# 1. THE VIBE-CHECKER (Sentiment Analysis)
# =============================================
# This is like borrowing a **pro surfer's brain** to judge if a comment is **gnarly** (good) or **bogus** (bad).
# In JS, this would be like using a `filter()` function to sort good vibes from bad ones.

print("🏄‍♂️ Downloading the Vibe-Checker... wait for it, brah...")
# `pipeline("sentiment-analysis")` is like calling a **lifeguard** to check the vibe.
# It's a pre-trained AI model that already knows how to judge text.
vibe_checker = pipeline("sentiment-analysis")

# =============================================
# 2. THE TEST (Let's Check Some Comments)
# =============================================
# We're gonna test the Vibe-Checker with some **rad** and **bogus** comments.
# Think of this like testing a new surfboard—you gotta see if it handles the waves right.

comments = [
    "Brah, that 360 air was absolutely gnarly! 🤙",  # This should be **stoked** (positive).
    "Your code is mid and your layout is bogus. L.",  # This should be **sucks** (negative).
    "I'm lowkey stoked about this pizza."  # This should be **stoked** (positive).
]

print("\n--- VIBE-CHECK RESULTS ---")
for comment in comments:
    # `vibe_checker(comment)` is like asking the lifeguard: "Is this vibe gnarly or bogus?"
    result = vibe_checker(comment)
    print(f"Comment: {comment}")
    print(f"AI Vibe: {result[0]['label']} (Score: {result[0]['score']:.2f})")
    # The `score` is like how **confident** the lifeguard is—closer to 1.0 means "deadset gnarly."

    # JS Analogy:
    # This is like `console.log(filteredComments)` where `filteredComments` is an array of vibes.
    print()  # Empty line for readability (like a space between waves).

# =============================================
# 3. THE STORYTELLER (Text Generation)
# =============================================
# Now we're gonna make the AI **write a surf story** like a **shaka master**.
# Think of this like feeding a prompt to a **story generator**—the AI fills in the blanks.

print("📖 Downloading the Storyteller... wait for it, dude...")
# `pipeline("text-generation", model="gpt2")` is like borrowing a **shaka master's brain** to write a story.
# `gpt2` is a smaller version of the AI that powers big models like ChatGPT.
generator = pipeline("text-generation", model="gpt2")

prompt = "The surfer dropped into a 10-foot wave and"
# This is like giving the shaka master a **starting line**—they'll take it from here.

print("\n--- AI STORY TIME ---")
# `generator(prompt, max_length=30, num_return_sequences=1)` is like saying:
# "Shaka master, write a story starting with this prompt, but keep it short (30 words max)."
story = generator(prompt, max_length=30, num_return_sequences=1)
print(story[0]['generated_text'])

# JS Analogy:
# This is like `storyGenerator.generate(prompt)` where `storyGenerator` is a function that writes stories.

# =============================================
# 4. THE CHALLENGE (Student Tasks)
# =============================================
# Now it's your turn to **shred** with this AI brain! Try these exercises:

print("\n--- STUDENT CHALLENGES ---")
print("1. Run the Vibe-Checker. Does it understand 'Gnarly' is good?")
print("   (Hint: The AI was trained on normal text, so 'gnarly' might confuse it. It's like teaching a kook to surf—it takes time!)")

print("\n2. The Experiment: Change the `max_length` in the story generator.")
print("   Can you make it write a whole essay?")
print("   (Hint: Try `max_length=100` or `max_length=200`. It's like letting the shaka master go wild—sometimes they write **cooked** stories, sometimes they write **bogus** ones.)")

print("\n3. The Roast: Try to make the AI say something 'Sus'.")
print("   (It won't—it’s been trained to stay chill. It's like a lifeguard who never panics, even in a **gnarly** wipeout.)")

# =============================================
# 5. SUMMARY (The Big Picture)
# =============================================
# You just used the **same tech** that powers the biggest AIs in the world.
# - **Attention** is the **ultimate flex**—it's how the AI focuses on the right words (like a surfer focusing on the right wave).
# - **Transformers** are like **pro surfers** who can handle **any wave** (text, images, code, etc.).
# - **Slay.** (That's it. You're now a **shaka master** of AI.)

print("\n🤙 Summary: You just used the same tech that powers the biggest AIs in the world. Attention is the ultimate flex. Slay.")