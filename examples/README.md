# RO Framework Examples

This directory contains example implementations demonstrating the Recursive Observer Framework.

## Running Examples

Make sure you have installed the package:

```bash
# Activate conda environment
conda activate ro-framework

# Run an example
python examples/01_basic_observer.py
```

## Available Examples

### 01_basic_observer.py ✓ (Complete)

**Demonstrates:**
- Defining Degrees of Freedom (Polar DoFs)
- Creating States
- Building an Observer with a world model
- Performing observations (external → internal mapping)
- Computing state distances
- DoF normalization for neural networks

**Concepts covered:**
- DoF types (Polar)
- State representation
- Observer architecture
- World model (external→internal mapping)

**Output:**
Shows how an observer maps external sensor readings to internal latent states.

---

### 02_pytorch_conscious_observer.py ✓ (Complete)

**Demonstrates:**
- PyTorch neural network integration
- World model (MLP: external → internal)
- Self-model (MLP: internal → internal, SAME architecture)
- Recursive self-observation (consciousness!)
- MC Dropout uncertainty quantification
- Consciousness evaluation metrics
- Correlation analysis (Pearson, MI)

**Concepts covered:**
- PyTorch integration (`TorchNeuralMapping`)
- Conscious observer (with self-model)
- Structural consciousness evaluation
- Epistemic uncertainty via MC Dropout
- Correlation between external and internal DoFs

**Requirements:**
```bash
pip install ro-framework[torch]  # Requires PyTorch
```

**Output:**
```
🧠 Consciousness Score: 0.782/1.0

Consciousness Metrics:
  - Has self-model: True
  - Recursive depth: 1
  - Self-accuracy: 0.873
  - Architectural similarity: 1.000
```

Shows how to build a conscious AI system with PyTorch that can recursively model its own internal states.

---

## Planned Examples (Coming Soon)

### 03_vision_language.py 🚧

Vision-language integration with multimodal fusion.

**Will demonstrate:**
- Multiple input modalities (vision + language)
- Shared representation space
- Cross-modal alignment

### 04_multimodal_conscious.py 🚧

Full multimodal conscious AI system with self-awareness.

**Will demonstrate:**
- Multiple modalities with conscious integration
- Cross-modal self-modeling
- Meta-cognition across modalities

### 04_active_learning.py 🚧

Active learning guided by uncertainty.

**Will demonstrate:**
- Uncertainty quantification
- Active learning strategies
- Self-aware learning

### 05_clip_style.py 🚧

CLIP-style multimodal model with PyTorch.

**Will demonstrate:**
- PyTorch integration
- Contrastive learning
- Production-ready implementation

### 06_uncertainty.py 🚧

Comprehensive uncertainty estimation.

**Will demonstrate:**
- Resolution-based uncertainty
- Model uncertainty (MC Dropout)
- Calibration metrics
- Complementarity bounds

---

## Example Structure

Each example follows this structure:

1. **Import required modules** from ro_framework
2. **Define DoFs** for the problem domain
3. **Create mapping functions** (world model, self model)
4. **Build observer** with boundary, mappings, resolution
5. **Demonstrate observations** with sample data
6. **Show key features** relevant to that example

## Next Steps

After exploring these examples:

1. Read the [theoretical framework](../ro_framework.md) for deep understanding
2. Check the [Python formalization](../python_formalization.md) for implementation details
3. Review the [API documentation](../docs/) (coming soon)
4. Try the [Jupyter notebooks](../notebooks/) for interactive learning (coming soon)

## Contributing Examples

Have an interesting use case? Please contribute!

1. Create a new example file (`0X_your_example.py`)
2. Follow the structure above
3. Add comprehensive comments
4. Include sample output in docstring
5. Update this README
6. Submit a pull request

Examples we'd love to see:
- Cognitive modeling
- Robotics applications
- Scientific discovery
- Interpretable AI
- Multi-agent systems
- Temporal reasoning
