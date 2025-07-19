# FreqNet: Hybrid Deepfake Detection Framework

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

**FreqNet** is a cutting-edge hybrid Deepfake detection model that combines deep learning with classical image processing to identify synthetic media with high accuracy and robustness. By fusing ResNet-50’s semantic features with handcrafted descriptors (FFT, Canny, LBP), FreqNet detects subtle artifacts missed by traditional CNNs, making it ideal for digital forensics, social media moderation, and law enforcement.

## Project Overview

Deepfakes are synthetic media mimicking real human appearances that threaten media authenticity, enabling misinformation and fraud. Traditional CNN-based detectors (e.g., ResNet-50, XceptionNet) focus on semantic cues but struggle with statistical artifacts like frequency inconsistencies or unnatural textures. **FreqNet** addresses this by analyzing *how* Deepfakes are made, not just *what* they depict, achieving superior performance across diverse manipulation types.

## Problem Statement

Deepfakes, powered by GANs, VAEs, and face-swapping tools, create realistic media that challenge trust in digital content. While CNNs achieve high accuracy in controlled settings, they falter with:
- Subtle statistical artifacts (e.g., frequency noise, edge smoothing).
- Generalization to unseen manipulations or compressed media.
- Lack of interpretability in forensic applications.

FreqNet overcomes these limitations with a hybrid approach, combining deep and handcrafted features for robust, interpretable detection.

## Novelty of FreqNet

FreqNet redefines Deepfake detection with a **hybrid architecture** that sets it apart:
- **Feature Fusion**: Integrates ResNet-50’s deep semantic features with handcrafted descriptors:
  - **Fast Fourier Transform (FFT)**: Detects frequency-domain anomalies.
  - **Canny Edge Detection**: Captures unnatural edge smoothness.
  - **Local Binary Patterns (LBP)**: Identifies texture irregularities.
- **Superior Performance**: Achieves 91.6% accuracy and 0.948 AUC, outperforming baseline CNNs by 7.4% in accuracy.
- **Robustness**: Excels in challenging conditions (e.g., compression, low resolution, unseen fakes).
- **Interpretability**: Handcrafted features provide clear insights into *why* an image is fake, enhancing trust in forensic use cases.

## Results

FreqNet was evaluated on a balanced Deepfake dataset from [Kaggle](https://www.kaggle.com/datasets/manj11karki/deapraka-and-real-images), outperforming a baseline CNN across key metrics:

| Model           | Accuracy | AUC   | Precision | Recall |
|-----------------|----------|-------|-----------|--------|
| Baseline CNN     | 84.2%    | 0.881 | 0.83      | 0.81   |
| **FreqNet**     | **91.6%** | **0.948** | **0.90** | **0.93** |

**Highlights**:
- **7.4% accuracy boost** and **0.067 AUC improvement** over the baseline.
- Detects subtle artifacts (e.g., unnatural textures, edge inconsistencies) missed by CNNs.
- Reduces false positives and negatives, critical for real-world applications.

## Demo Video

Watch FreqNet in action! Our deployment demo showcases real-time Deepfake detection on sample images and videos.

🔗 [View Demo Video](https://drive.google.com/file/d/1-gnwkmd9704hmey6Sh5fnnPhhjz8-LUU/view?usp=drive_link) 

##  Implementation

- **Framework**: PyTorch (GPU-accelerated).
- **Architecture**: Hybrid model with frozen ResNet-50 (up to layer 4) and handcrafted feature extraction (FFT, Canny, LBP).
- **Training**:
  - Images: 256x256, normalized with ImageNet statistics.
  - Optimizer: Adam (LR=1e-4, weight decay=1e-5).
  - Scheduler: ReduceLROnPlateau (patience=3, factor=0.5).
  - Early Stopping: Patience=5.
  - Loss: Binary cross-entropy.
- **Dataset**: Balanced Deepfake dataset with real and fake images from diverse synthesis pipelines.

## Deployment

FreqNet is designed for real-world deployment:
- **Use Case**: Social media content moderation, digital forensics, law enforcement.
- **Setup**: Deployable as a standalone Python application or integrated into web platforms via Steamliit.
-  Deployed on [Streamlit]() 



## Limitations

- **Dataset Scope**: Evaluated on a single dataset; may require cross-dataset validation.
- **Computational Cost**: Handcrafted feature extraction adds overhead, impacting real-time performance.
- **Frame-Based**: Lacks temporal analysis for video Deepfakes.

## Future Work

- **Cross-Dataset Testing**: Evaluate on FaceForensics++, DFDC, Celeb-DF for broader generalizability.
- **Video Analysis**: Add temporal features (e.g., optical flow, LSTMs) for video Deepfakes.
- **Optimization**: Use lightweight models (e.g., MobileNet) or pruning for faster inference.
- **Explainability**: Integrate Grad-CAM/SHAP for transparent predictions.
- **Adversarial Defense**: Enhance robustness against attacks targeting Deepfake detectors.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd freqnet
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Dataset**:
   - Get the Deepfake dataset from [Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images).
4. **Train the Model**:
   ```bash
   python train.py --dataset_path <path-to-dataset> --model_path <path-to-save-model>
   ```
5. **Evaluate**:
   ```bash
   python evaluate.py --model_path <path-to-trained-model> --test_data <path-to-test-data>
   ```
6. **Run Deployment**:
   ```bash
   python deploy.py --model_path <path-to-trained-model> --input <image-or-video-path>
   ```

## 👥 Author

- **Aruni Saxena** (202418006)

## 📜 Citation

```bibtex
@article{saxena2025freqnet,
  title={Deepfake Forensics Reimagined: Learning How It Was Faked, Not Just What Is Fake},
  author={Saxena, Aruni},
