# Human versus ViT

A comparative study examining the performance differences between human visual perception and Vision Transformer (ViT) models in image recognition and classification tasks.

## Overview

This repository contains the implementation and analysis code for comparing human cognitive abilities with Vision Transformer neural networks. The project explores how humans and state-of-the-art computer vision models perform on various visual recognition tasks, providing insights into the strengths and limitations of both biological and artificial vision systems.

## Research Objectives

- **Performance Comparison**: Quantitative analysis of human vs. ViT performance across different image classification tasks
- **Robustness Analysis**: Evaluation of both systems under various image distortions and challenging conditions
- **Cognitive Insights**: Understanding the fundamental differences in how humans and transformers process visual information
- **Benchmark Development**: Creating standardized evaluation protocols for human-AI comparison studies

## Key Features

### Experimental Framework
- Controlled experimental setup for human subject testing
- Standardized evaluation protocols for ViT models
- Statistical analysis tools for performance comparison
- Comprehensive data collection and processing pipeline

### Model Implementations
- Pre-trained Vision Transformer models (ViT-Base, ViT-Large)
- Fine-tuning capabilities for domain-specific tasks
- Support for various ViT architectures and configurations
- Integration with popular deep learning frameworks

### Human Study Components
- Psychophysical experiment design
- Response time measurement
- Accuracy assessment protocols
- User interface for human testing sessions

## Dataset Support

The framework supports evaluation on multiple standard datasets:
- **ImageNet**: Large-scale image classification
- **CIFAR-10/100**: Small-scale natural image classification
- **Custom Datasets**: Domain-specific evaluation sets
- **Distorted Images**: Robustness testing with various image corruptions

## Installation

### Requirements
```bash
Python >= 3.8
PyTorch >= 1.9.0
torchvision >= 0.10.0
transformers >= 4.0.0
numpy >= 1.21.0
matplotlib >= 3.3.0
scipy >= 1.7.0
pandas >= 1.3.0
```

### Setup
```bash
# Clone the repository
git clone https://github.com/mlacarrasco/human_versus_vit.git
cd human_versus_vit

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (optional)
python scripts/download_models.py
```

## Usage

### Running ViT Evaluation
```python
from src.models import ViTEvaluator
from src.datasets import load_dataset

# Initialize evaluator
evaluator = ViTEvaluator(model_name='vit_base_patch16_224')

# Load dataset
dataset = load_dataset('imagenet', split='validation')

# Run evaluation
results = evaluator.evaluate(dataset)
print(f"ViT Accuracy: {results['accuracy']:.2f}%")
```

### Human Study Interface
```bash
# Launch human study interface
python human_study/run_experiment.py --config configs/human_study.yaml
```

### Comparative Analysis
```python
from src.analysis import compare_performance

# Load results from both human and ViT evaluations
human_results = load_human_results('data/human_results.json')
vit_results = load_vit_results('data/vit_results.json')

# Generate comparison
comparison = compare_performance(human_results, vit_results)
comparison.plot_results()
```

## Experimental Protocol

### Human Study Design
1. **Participant Recruitment**: Controlled selection criteria for human subjects
2. **Task Design**: Standardized image classification tasks with time constraints
3. **Data Collection**: Systematic recording of responses and reaction times
4. **Quality Control**: Validation protocols to ensure data reliability

### ViT Evaluation Protocol
1. **Model Selection**: Systematic evaluation across different ViT architectures
2. **Preprocessing**: Standardized image preprocessing pipeline
3. **Inference**: Controlled evaluation environment with consistent parameters
4. **Performance Metrics**: Comprehensive accuracy and efficiency measurements

## Results Structure

```
results/
├── human_studies/
│   ├── raw_data/
│   ├── processed/
│   └── analysis/
├── vit_evaluation/
│   ├── model_outputs/
│   ├── performance_metrics/
│   └── ablation_studies/
└── comparative_analysis/
    ├── statistical_tests/
    ├── visualizations/
    └── reports/
```

## Key Findings

### Performance Metrics
- **Overall Accuracy**: Comparative analysis across different image categories
- **Response Time**: Speed comparison between human cognition and model inference
- **Robustness**: Performance degradation under various image distortions
- **Category-specific Performance**: Detailed analysis of performance across object categories

### Statistical Analysis
- **Significance Testing**: Statistical validation of performance differences
- **Correlation Analysis**: Relationship between human and ViT performance patterns
- **Error Analysis**: Systematic examination of failure modes in both systems

## Configuration

### Model Configuration
```yaml
# config/vit_config.yaml
model:
  name: "vit_base_patch16_224"
  pretrained: true
  num_classes: 1000
  
evaluation:
  batch_size: 32
  num_workers: 4
  device: "cuda"
```

### Human Study Configuration
```yaml
# config/human_study.yaml
experiment:
  num_participants: 50
  trials_per_participant: 200
  time_limit: 2.0  # seconds
  
display:
  image_size: [224, 224]
  presentation_time: 1.0
  inter_trial_interval: 0.5
```

## Contributing

We welcome contributions to improve the experimental framework and analysis tools. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Vision Transformer implementation based on the original paper by Dosovitskiy et al.
- Human psychophysical experiment design inspired by cognitive psychology literature
- Statistical analysis methods adapted from comparative psychology research
- Special thanks to all human participants in the study

## Contact

**Miguel Carrasco**  
- Email: [contact information]
- Website: https://mlacarrasco.github.io/
- LinkedIn: [profile link]

For questions about the research methodology or experimental design, please open an issue or contact the author directly.

## Related Work

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Model-vs-Human: Benchmark your model on out-of-distribution datasets](https://github.com/bethgelab/model-vs-human)
- [Vision Transformer for Generic Body Pose Estimation](https://github.com/ViTAE-Transformer/ViTPose)
- [Human-like ViTs: ViT models pretrained with human-like video data](https://github.com/eminorhan/humanlike-vits)
