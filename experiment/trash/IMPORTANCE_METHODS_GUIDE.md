# Channel Importance Calculation Methods Guide

This guide explains the 7 different methods available for calculating channel importance in our explainability analysis.

## 🎯 Quick Start

To change the importance calculation method, edit line 646 in `channel_noise_sensitivity_road.py`:

```python
IMPORTANCE_METHOD = 'combined'  # <-- Change this value
```

---

## 📊 Available Methods

### 1️⃣ **`'gradient'`** - Gradient Magnitude
```python
importance = gradients.abs().mean(dim=(0, 2, 3))
```

**What it measures:**
- How sensitive the output is to changes in each channel
- Channels with high gradients have strong influence on predictions

**Interpretation:**
- High value → Output is very sensitive to this channel
- Low value → Output barely changes when this channel changes

**Use when:**
- You want to find channels that have the strongest impact on model decisions
- Focus on sensitivity-based importance

---

### 2️⃣ **`'activation'`** - Activation Magnitude
```python
importance = activations.abs().mean(dim=(0, 2, 3))
```

**What it measures:**
- The average firing strength of each channel
- How "active" or "responsive" each channel is

**Interpretation:**
- High value → Channel fires strongly on average
- Low value → Channel is mostly inactive

**Use when:**
- You want to identify channels that are most active
- Focus on signal strength rather than sensitivity

---

### 3️⃣ **`'weight'`** - GradCAM Weights (α_k)
```python
importance = weights.abs().mean(dim=(0, 2, 3))
# where weights = gradients.mean(dim=(2, 3))
```

**What it measures:**
- The GradCAM weights (α_k) from the original GradCAM formula
- Global average pooling of gradients

**Interpretation:**
- High value → Channel has strong positive/negative influence on target class
- This is the "pure" GradCAM importance metric

**Use when:**
- You want to stay close to the original GradCAM definition
- Focus on class-specific importance

---

### 4️⃣ **`'combined'`** - Weight × Activation (RECOMMENDED ⭐)
```python
importance = (weights.abs() * activations.abs()).mean(dim=(0, 2, 3))
```

**What it measures:**
- The actual weighted contribution to the GradCAM heatmap
- Combines both sensitivity (gradient) and strength (activation)

**Interpretation:**
- High value → Channel contributes strongly to the final explanation
- Balances "how important" (weight) with "how active" (activation)

**Use when:**
- **Default choice for most experiments**
- You want a balanced measure of importance
- You care about actual contribution to explainability

**Why it's recommended:**
- Most directly related to GradCAM output
- Considers both gradient and activation
- Empirically works well in practice

---

### 5️⃣ **`'gradcam_contribution'`** - Actual CAM Contribution
```python
weighted_activation = weights * activations
cam_contribution = weighted_activation.mean(dim=(2, 3))
importance = F.relu(cam_contribution).mean(dim=0)
```

**What it measures:**
- The exact per-channel contribution to the final GradCAM after ReLU
- Accounts for the ReLU operation in GradCAM

**Interpretation:**
- High value → Channel positively contributes to the final heatmap
- Only considers positive contributions (negative values zeroed out)

**Use when:**
- You want to measure the exact contribution to the visual heatmap
- You care about post-ReLU importance
- You want to exclude channels with negative contributions

---

### 6️⃣ **`'taylor'`** - Taylor Approximation
```python
importance = (gradients * activations).abs().mean(dim=(0, 2, 3))
```

**What it measures:**
- First-order Taylor approximation of output change
- Estimates: Δoutput ≈ gradient × activation

**Interpretation:**
- High value → Removing this channel would significantly change the output
- Approximates the counterfactual "what if this channel was zero?"

**Use when:**
- You want to estimate the effect of removing channels
- Focus on counterfactual importance
- Similar to "integrated gradients" concept

---

### 7️⃣ **`'variance'`** - Activation Variance
```python
mean_act = activations.mean(dim=(2, 3), keepdim=True)
importance = ((activations - mean_act) ** 2).mean(dim=(0, 2, 3))
```

**What it measures:**
- How much each channel's activation varies spatially
- High variance = more selective/informative channel

**Interpretation:**
- High value → Channel responds very differently to different spatial locations
- Low value → Channel has uniform activation (less selective)

**Use when:**
- You want to find channels that are spatially selective
- Focus on information content rather than magnitude
- Interested in feature selectivity

---

## 🔬 Comparison Table

| Method | Type | Gradient | Activation | Spatial | Best For |
|--------|------|----------|------------|---------|----------|
| `gradient` | Sensitivity | ✅ | ❌ | Global | Finding sensitive channels |
| `activation` | Magnitude | ❌ | ✅ | Global | Finding active channels |
| `weight` | GradCAM | ✅ | ❌ | Global | Original GradCAM importance |
| **`combined`** ⭐ | Hybrid | ✅ | ✅ | Global | **Recommended default** |
| `gradcam_contribution` | Exact CAM | ✅ | ✅ | Global | Exact heatmap contribution |
| `taylor` | Counterfactual | ✅ | ✅ | Global | Removal impact estimation |
| `variance` | Selectivity | ❌ | ✅ | Local | Feature selectivity |

---

## 💡 Recommendation Guide

### For General Explainability Analysis:
→ Use **`'combined'`** (default)

### For Quantization/Compression:
→ Use **`'combined'`** or **`'gradcam_contribution'`**

### For Sensitivity Analysis:
→ Use **`'gradient'`** or **`'taylor'`**

### For Channel Pruning:
→ Use **`'activation'`** or **`'variance'`**

### For Pure GradCAM Research:
→ Use **`'weight'`** or **`'gradcam_contribution'`**

---

## 🧪 Example: Running with Different Methods

```python
# In channel_noise_sensitivity_road.py, line 646:

# Test 1: Default (recommended)
IMPORTANCE_METHOD = 'combined'

# Test 2: Pure gradient-based
IMPORTANCE_METHOD = 'gradient'

# Test 3: Activation-based
IMPORTANCE_METHOD = 'activation'

# Test 4: Exact CAM contribution
IMPORTANCE_METHOD = 'gradcam_contribution'
```

Each method will generate a separate output file:
- `results/channel_noise_sensitivity_road_combined.png`
- `results/channel_noise_sensitivity_road_gradient.png`
- `results/channel_noise_sensitivity_road_activation.png`
- etc.

---

## 📈 Expected Differences

Different methods may identify different "top channels":

- **`gradient`**: Channels with high sensitivity to output
- **`activation`**: Channels that fire strongly
- **`combined`**: Balanced view (usually best)
- **`variance`**: Channels with selective responses

**Impact on results:**
- The ranking of Top/Mid/Bottom channels may change
- The noise sensitivity curves may shift
- But the general trend should remain consistent

---

## 🎓 Theoretical Background

### GradCAM Formula:
```
L^c_GradCAM = ReLU(Σ_k α_k^c · A_k)

where:
  α_k^c = (1/Z) Σ_i Σ_j (∂y^c / ∂A_k^{ij})  [weight method]
  A_k = activation of channel k              [activation method]
```

### Our Methods in Context:
- `weight`: Uses α_k^c
- `activation`: Uses A_k
- `combined`: Uses α_k^c × A_k (before summation)
- `gradcam_contribution`: Uses ReLU(α_k^c × A_k) (after ReLU)

---

## ✅ Validation

To verify your importance method is working correctly:

```python
# Check that importance scores sum to reasonable values
print(f"Min importance: {importance_scores.min():.4f}")
print(f"Max importance: {importance_scores.max():.4f}")
print(f"Mean importance: {importance_scores.mean():.4f}")

# Check distribution
import matplotlib.pyplot as plt
plt.hist(importance_scores.numpy(), bins=50)
plt.title(f"Importance Distribution ({IMPORTANCE_METHOD})")
plt.show()
```

---

**Author:** Channel Noise Sensitivity Analysis Framework
**Date:** 2025-10-30
**Version:** 1.0
