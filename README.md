# 🐶 Pre-trained Image Classifier — Comparing CNN Architectures for Dog Breed Classification

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/Udacity-AI_Programming_Nanodegree-02B3E4)

## 🚀 Overview

A command-line application that compares three pre-trained CNN architectures (VGG, ResNet, AlexNet) on their ability to classify dog breeds and distinguish dogs from non-dogs. The pipeline extracts ground-truth labels from filenames, runs inference through each model, cross-references against a 223-breed dog name dictionary, and computes classification statistics. VGG achieves the best overall performance: 100% dog detection, 93.3% breed accuracy, and 100% non-dog classification.

## 📊 Results — Model Comparison

| Metric | VGG | ResNet | AlexNet |
|---|---|---|---|
| % Correct Dogs | **100%** | **100%** | **100%** |
| % Correct Breed | **93.3%** | 90.0% | 80.0% |
| % Correct Not-Dogs | **100%** | 90.0% | **100%** |
| % Label Match | **87.5%** | 82.5% | 75.0% |
| Runtime | 32s | 5s | 3s |

**Key Finding:** VGG has the highest breed accuracy (93.3%) but is the slowest (32s). ResNet misclassified a cat as a dog. AlexNet is fastest but has the lowest breed accuracy (80%).

**Common Misclassifications:**
- Great Pyrenees → Kuvasz (all 3 models)
- Beagle → Walker Hound (all 3 models)
- Golden Retriever → Leonberg/Afghan Hound (ResNet, AlexNet)

## ✨ Key Features

- **Three-Model Benchmarking** — Runs VGG, ResNet, and AlexNet on the same dataset via a single batch script, piping results to text files for comparison
- **Filename-Based Ground Truth** — Pet labels are extracted from image filenames (e.g., `Beagle_01125.jpg` → `beagle`), eliminating the need for a separate labels file
- **Dog vs. Not-Dog Classification** — Uses ImageNet class indices to determine if the classifier output is a dog breed, separate from breed accuracy
- **223-Breed Dictionary** — `dognames.txt` contains all recognized dog breed names for cross-referencing classifier output
- **Misclassification Reporting** — Prints both incorrect dog/not-dog assignments and incorrect breed assignments with pet label vs. classifier label

## 🧠 Technical Highlights

- **Modular Pipeline** — 6 Python modules chained together: `get_input_args` → `get_pet_labels` → `classify_images` → `adjust_results4_isadog` → `calculates_results_stats` → `print_results`
- **Results Dictionary Structure** — Each image maps to a 5-element list: `[pet_label, classifier_label, label_match(0/1), is_dog(0/1), classifier_is_dog(0/1)]`, enabling flexible statistics computation
- **Statistics Computation** — Counts and percentages calculated from the results dictionary: n_images, n_dogs_img, n_correct_dogs, n_correct_breed, n_correct_notdogs, with percentage derivations
- **Batch Execution** — `run_models_batch.sh` runs all three architectures sequentially, piping output to `{model}_pet-images.txt` for offline comparison

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Framework | PyTorch (torchvision pre-trained models) |
| Models | VGG, ResNet, AlexNet (ImageNet pre-trained) |
| CLI | argparse (--dir, --arch, --dogfile) |
| Dataset | 40 pet images (30 dogs, 10 non-dogs) + 4 uploaded test images |
| Dog Names | 223 breeds from `dognames.txt` |

## 🏗 Pipeline

```
Image Files (filename = ground truth label)
    │
    ▼
get_pet_labels() → Extract breed from filename
    │
    ▼
classify_images() → Run CNN (vgg/resnet/alexnet) → classifier label
    │
    ▼
adjust_results4_isadog() → Check if pet/classifier labels are dog breeds
    │
    ▼
calculates_results_stats() → Compute counts & percentages
    │
    ▼
print_results() → Summary + misclassified dogs + misclassified breeds
```

## ⚡ Getting Started

```bash
git clone https://github.com/jashjain21/Pre-trained-Image-Classifier-to-classify-Dog-Breeds.git
cd Pre-trained-Image-Classifier-to-classify-Dog-Breeds

pip install torch torchvision pillow

# Run all three models
sh run_models_batch.sh

# Run a single model
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt

# Test with your own images
sh run_models_batch_uploaded.sh
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--dir` | `pet_images/` | Directory containing images to classify |
| `--arch` | `vgg` | CNN architecture: `vgg`, `resnet`, or `alexnet` |
| `--dogfile` | `dognames.txt` | Text file with recognized dog breed names |

## 🔍 What This Project Demonstrates

- **Model Comparison** — Systematic benchmarking of 3 CNN architectures on the same dataset with quantitative metrics (accuracy, runtime)
- **Classification Pipeline Design** — Modular, 6-stage pipeline where each stage has a single responsibility and passes data via a shared results dictionary
- **Practical Trade-offs** — VGG is most accurate but 10× slower than AlexNet; ResNet balances speed and accuracy but misclassifies non-dogs — demonstrating real-world model selection considerations
- **CLI Application** — Configurable command-line tool with argparse and batch scripts for reproducible experiments

## 🚧 Limitations / Future Improvements

- **Small Test Set** — Only 40 images (30 dogs, 10 non-dogs); a larger test set would give more statistically significant results
- **No Confidence Thresholds** — The classifier's top-1 prediction is used regardless of confidence; filtering by probability would reduce false positives
- **Filename-Based Labels** — Ground truth depends on correct filenames; a proper label file or directory structure would be more robust
- **No Fine-Tuning** — Models are used as-is from ImageNet; fine-tuning on a dog breed dataset would improve breed-level accuracy

## Author
Jash Jain : [LinkedIn](https://www.linkedin.com/in/jash-jain-bb659a132)
