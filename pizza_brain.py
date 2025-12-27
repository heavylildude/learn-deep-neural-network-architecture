import numpy as np  

### **1. THE DATA (Clues)**

# Pizza features: [Cheesiness, Crunchiness, Greasiness]
# JS analogy: `const pizzaClues = [0.9, 0.8, 0.2];`
pizza_clues = np.array([0.9, 0.8, 0.2])
# Think of this like a **pizza report card**—how much of each trait it has.

### **2. THE RIZZ (Weights)**

# Weights = How much we **care** about each trait.
# JS analogy: `const rizzFactors = [10.0, 8.0, -5.0];`
rizz_factors = np.array([10.0, 8.0, -5.0])
# - **Cheese (10.0)**: We **fucking love** cheese → high positive weight.
# - **Crunch (8.0)**: Crunch is **rad** → positive but less than cheese.
# - **Grease (-5.0)**: Grease is **bogus** → negative weight (we **hate** it).

### **3. THE MOOD (Bias)**

# Bias = Our **baseline hunger level** (how easy we are to impress).
# JS analogy: `const mood = 2.0;`
mood = 2.0
# Think of this like our **default "yes"**—even a **basic** pizza starts with +2 points.

### **4. THE ACTIVATION (ReLU)**

def activation_relu(x):
    # ReLU = "If it's good, keep it. If it's shit, make it 0."
    # JS analogy: `Math.max(0, x)`
    return max(0, x)
    # - If `x` is **positive** → return `x` (we like it).
    # - If `x` is **negative** → return `0` (we **reject** it).
    
### **5. THE PIZZA BRAIN (Neural Network)**

def pizza_brain(clues, weights, bias):
    # Step 1: Calculate the **total vibe score** (dot product + bias).
    # JS analogy: `clues.reduce((sum, clue, i) => sum + clue * weights[i], bias)`
    total_vibe = np.dot(clues, weights) + bias
    print(f"Total Vibe Score: {total_vibe}")  # Debugging: "Yo, this pizza got {score} points!"

    # Step 2: Apply ReLU to decide if it's **bussin** or **basic**.
    result = activation_relu(total_vibe)
    return result

### **6. THE DECISION**
# Run the brain on our pizza clues.
score = pizza_brain(pizza_clues, rizz_factors, mood)

# If score > 15 → **BUSSIN** (we **order** it).
# If score <= 15 → **BASIC** (we **reject** it).
if score > 15:
    print("AI SAYS: BUSSIN! 🍕🔥")  # JS: `console.log("BUSSIN!")`
else:
    print("AI SAYS: BASIC... L 🤮")  # JS: `console.log("BASIC...")`