# Math of a Neural Net

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/marcoharuni/Learn-Neural-Net/blob/main/Math%20of%20a%20neural%20net.ipynb)

## Overview

This notebook is a step-by-step, mathematically rigorous walkthrough of the foundations of neural networks, focusing on **gradient descent** and the **math underlying a single neuron**. It is designed as an interactive and visual learning resource for beginners and those looking to understand neural nets from first principles—no frameworks or "black box" abstractions.

**Key topics:**
- Manual computation of gradients and weight updates
- Visualizing gradient descent dynamics
- Stability and convergence analysis
- Exploding gradients and their prevention
- Transition from scalars to vectors (multi-dimensional inputs)
- Practical coding with Python and NumPy

---

## Table of Contents

1. [Chapter 1: Math is the CORE](#chapter-1-math-is-the-core)
    - Step-by-step gradient descent for a single neuron
    - Manual computation of predictions, loss, gradients, and updates
2. [Chapter 2: Visualizing Weight Updates](#chapter-2-lets-plot-how-weight-w-changes)
    - Plotting convergence of weights during training
    - Oscillation vs. smooth convergence (role of learning rate)
3. [Exploding Gradients](#exploding-gradients)
    - What are exploding gradients?
    - Code and plots illustrating divergence
    - How to prevent instability
    - Mathematical and geometric analysis of convergence
4. [Vectors: Multi-dimensional Neurons](#vectors)
    - Extending single neuron math to vector inputs and weights
    - Manual forward and backward pass with vectors

---

## 1. Chapter 1: Math is the CORE

The notebook starts with a **simple linear neuron**:  
$$ y_\text{pred} = w \cdot x $$
with squared loss:  
$$ L = (y_\text{pred} - y_\text{true})^2 $$

**Tasks Covered:**
- Predict output given $x$, $w$
- Calculate error and loss
- Compute gradient $\frac{dL}{dw}$ using the chain rule
- Update $w$ with gradient descent:  
  $$ w_{\text{new}} = w - \eta \frac{dL}{dw} $$
- Repeat updates to observe convergence

**Detailed solution steps** are included for each update, making the math explicit and accessible.

---

## 2. Chapter 2: Let's plot how weight (W) changes

- **Visualization:** The notebook uses Matplotlib to plot the evolution of the weight $w$ across training steps.
- **Oscillation vs. Convergence:**  
  - Shows how a large learning rate can cause $w$ to oscillate around the minimum.
  - Demonstrates that a smaller learning rate leads to smooth, one-sided convergence.
- **Code included** for generating and customizing the plots.

---

## 3. Exploding Gradients

- **What it is:** Explains and visualizes how, with certain parameter choices, the gradient can become excessively large, causing weights to diverge (go to infinity or NaN).
- **Code experiments:**  
  - Try different values of $x$ and learning rates to see stable, critical, and divergent behavior.
  - Plots the regions of stability vs. instability for various learning rates and input magnitudes.
- **Mathematical Insight:**  
  - Derives the **convergence factor** and the critical condition for the learning rate:  
    $$ \text{For stability:} \quad \eta < \frac{1}{2x^2} $$
  - Discusses how this relates to input magnitude and feature scaling.
- **Prevention:**  
  - Suggests gradient clipping, better initialization, normalization, and architecture changes for deep nets.

---

## 4. Vectors

- **Moving beyond scalars:**  
  - Demonstrates how the same principles apply when $x$ and $w$ are vectors.
  - Shows the forward and backward pass for a neuron with multi-dimensional input:
    $$ y_\text{pred} = \vec{w}^\top \vec{x} $$
    $$ \vec{w}_{\text{new}} = \vec{w} - \eta \frac{\partial L}{\partial \vec{w}} $$
- **Manual computation** makes the vector calculus transparent.

---

## How to Use

1. **Open in Google Colab:**  
   Click the badge above or [open directly here](https://colab.research.google.com/github/marcoharuni/Learn-Neural-Net/blob/main/Math%20of%20a%20neural%20net.ipynb).

2. **Run All Cells:**  
   Step through the notebook cell by cell, reading the explanations and running the code. Visualizations will be rendered inline.

3. **Modify and Experiment:**  
   - Change input values, weights, learning rates.
   - Try different initializations and observe how convergence and stability change.

4. **Extend:**  
   - Try adding more dimensions to the vectors.
   - Implement batch gradient descent.
   - Add nonlinearity (activation functions) for further exploration.

---

## Prerequisites

- Python 3.x
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Jupyter](https://jupyter.org/) or Google Colab

No prior experience with deep learning frameworks is required. All math and code are built from scratch.

---

## Educational Goals

- Understand the **fundamental math of neural nets** without abstraction.
- Build intuition for **gradient descent** and its behavior.
- Learn how **learning rate** and **input scaling** affect convergence and stability.
- See practical examples of issues like **exploding gradients** and remedies.

---

## License

This notebook and its code are provided for educational and research purposes.  
See [LICENSE](LICENSE) for details.

---

## Author

Developed by [@marcoharuni](https://github.com/marcoharuni).

---

## Acknowledgments

Inspired by classic neural networks textbooks and courses.  
If you found this helpful, consider ⭐️ starring the repo!
