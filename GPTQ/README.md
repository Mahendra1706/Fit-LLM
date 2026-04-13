# GPTQ: The Math of LLM Quantization

This repository contains a complete mathematical breakdown of the **GPTQ** algorithm (Generalized Post-Training Quantization). 

**The Objective:** Compress massive, pre-trained Large Language Models (LLMs) into 4-bit weights so they can run on standard consumer GPUs, while using advanced second-order math to ensure the neural network doesn't lose its intelligence. 

Most implementations treat this as a black box. These notes strip away the code abstractions and prove the exact linear algebra and calculus happening under the hood, step-by-step.

---

## 1. The Setup: Calibration & The Hessian ($H$)

Before quantizing, we pass a high-quality calibration dataset (e.g., a subset of a 120-billion token dataset like C4) through the layer to observe how the weights interact with real data.

**The Covariance Heatmap:**
We calculate the Hessian ($H_F$), which in a pre-trained model simplifies to the covariance matrix of the input activations ($X_F$):

$$H_F = 2X_F X_F^T$$

This acts as a "heatmap" showing how often different inputs fire together. However, raw covariance is flawed—it double-counts indirect relationships. 

---

## 2. The Precision Matrix: Why we Inverse ($H^{-1}$)

If Weight A triggers Weight B, and Weight B triggers Weight C, the raw Hessian ($H$) will wrongly tell us that A and C are directly correlated. If we adjust C based on A's error, we will destroy the model.

To fix this, we take the mathematical inverse of the Hessian to create a **Precision Matrix** ($H^{-1}$). A precision matrix calculates *Partial Correlation*, which mathematically rips out the "middleman" to find the pure, direct relationship.

**The Math of Partial Correlation:**
$$\theta_{AC} \propto Cov(A,C) - \frac{Cov(A,B) \cdot Cov(B,C)}{Var(B)}$$

*The second part of the equation subtracts the exact contribution of B (the middleman). By using $H^{-1}$, we guarantee our compensation ratios are based on direct relationships only.*

---

## 3. The Loss Function: Newton's Method & Taylor Series

The GPTQ error function looks nothing like standard Cross-Entropy loss. Because the model is already pre-trained, it is mathematically resting at the exact, flat bottom of a loss valley.

When we force a weight to snap to a 4-bit grid, we introduce an error. Geometrically, we just pushed the model up the side of the valley. We need to know exactly how much the error increased, and **how far we need to jump to get back to the bottom.**

We use the **Taylor Series Quadratic Approximation** (Newton & Brook's equation) to calculate the new altitude:

$$\text{New Error} = (\text{Original Error}) + (\text{Slope} \times \delta) + \frac{1}{2}(\text{Curvature}) \times \delta^2$$

**Deleting the Zeros:**
1. **Original Error = 0**: We only care about the *increase* in error, so our baseline is 0.
2. **Slope = 0**: Because we are at the flat bottom of the valley, the first derivative (slope) is exactly 0.

Translated to matrix algebra, the exact error spike caused by quantization is simply:
$$E = \frac{1}{2} \delta^T H \delta$$

*Crucial Concept: In Newton's Method, $\delta$ is not a tiny learning step. It is the **exact, mathematically calculated distance** required to jump perfectly back to the lowest possible error state in one move.*

---

## 4. Lagrangian Mechanics: Finding the Exact Jump ($\delta$)

When we snap a target weight ($w_q$) to the 4-bit grid, it creates a raw error: $c$. 
We need to find the exact vector of adjustments ($\delta$) for all remaining weights to absorb this error.

**The Constraint:** We are legally forced to change the $q$-th weight by exactly $c$.
$$e_q^T \delta = c$$

To find the perfect jump ($\delta$) that minimizes $E$ while obeying the constraint, we build a **Lagrangian** ($\mathcal{L}$) with a stretching multiplier ($\lambda$):

$$\mathcal{L} = \frac{1}{2} \delta^T H \delta - \lambda (e_q^T \delta - c)$$

**The Derivation:**
Take the derivative with respect to $\delta$ and set it to $0$ to find the absolute minimum:
$$\frac{\partial \mathcal{L}}{\partial \delta} = H\delta - \lambda e_q = 0$$
$$\delta = \lambda H^{-1} e_q$$

To find the multiplier ($\lambda$), isolate the $q$-th column using our initial constraint:
$$\lambda [H^{-1}]_{qq} = c \implies \lambda = \frac{c}{[H^{-1}]_{qq}}$$

Substitute $\lambda$ back into the $\delta$ equation to get the final formula.

---

## 5. The Master Update Formula

$$\delta = -\frac{w_q - \text{quant}(w_q)}{[H^{-1}]_{qq}} \cdot (H^{-1})_{:,q}$$

**Breaking it down:**
* $w_q - \text{quant}(w_q)$: The Raw Error ($c$).
* $[H^{-1}]_{qq}$: The target weight's Self-Importance scaling factor (the diagonal).
* $(H^{-1})_{:,q}$: The exact compensation ratios for all other unquantized weights (the column).
* $\delta$: The final array of exact distances to jump.

---

## 6. The Cholesky Cheat Code: Killing Woodbury

The older OBQ algorithm recalculated the Inverse Hessian dynamically after every single weight using the **Woodbury** update formula:

$$H_{-q}^{-1} = H^{-1} - \frac{1}{[H^{-1}]_{qq}} H^{-1}_{:,q} (H^{-1}_{q,:})^T$$

**The Fatal Flaw:** Running this division-heavy update millions of times causes floating-point rounding errors to stack up. The matrix eventually slows down and blows up.

**The GPTQ Solution:** Instead of updating the matrix dynamically during the loop, GPTQ runs a **Cholesky Decomposition** on the Inverse Hessian *once*, before the loop even starts:

$$H^{-1} = L L^T$$

$L$ is a **Lower Triangular Matrix**. It is shaped like a staircase, where every number above the diagonal is exactly zero. 

These columns are perfectly pre-calculated Schur Complements. When we use the columns of $L$ to calculate our $\delta$ updates instead of $H^{-1}$, the zeros in the upper right of the matrix naturally act as shields. They mathematically ensure that the algorithm multiplies by zero and refuses to touch weights that have already been quantized and locked. 

It turns a mathematically unstable, snowballing loop into a blazing-fast, error-free spreadsheet lookup.