# FreqNet: Hybrid Deepfake Detection Framework

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

**FreqNet** is a cutting-edge hybrid Deepfake detection model that combines deep learning with classical image processing to identify synthetic media with high accuracy and robustness. By fusing ResNet-50’s semantic features with handcrafted descriptors—Fast Fourier Transform (FFT), Canny edge detection, and Local Binary Patterns (LBP)—FreqNet detects subtle artifacts missed by traditional CNNs, making it ideal for digital forensics, social media moderation, and law enforcement.

## 🚀 Project Overview

Deepfakes—synthetic media mimicking real human appearances—threaten media authenticity, enabling misinformation and fraud. Traditional CNN-based detectors (e.g., ResNet-50, XceptionNet) focus on semantic cues but struggle with statistical artifacts like frequency inconsistencies or unnatural textures. **FreqNet** addresses this by analyzing *how* Deepfakes are made, not just *what* they depict, achieving superior performance across diverse manipulation types.

## 🎯 Problem Statement

Deepfakes, powered by GANs, VAEs, and face-swapping tools, create realistic media that challenge trust in digital content. While CNNs achieve high accuracy in controlled settings, they falter with:
- Subtle statistical artifacts (e.g., frequency noise, edge smoothing).
- Generalization to unseen manipulations or compressed media.
- Lack of interpretability in forensic applications.

FreqNet overcomes these limitations with a hybrid approach, combining deep and handcrafted features for robust, interpretable detection.

## 🌟 Novelty of FreqNet

FreqNet redefines Deepfake detection with a **hybrid architecture** that sets it apart:
- **Feature Fusion**: Integrates ResNet-50’s deep semantic features with handcrafted descriptors:
  - **Fast Fourier Transform (FFT)**: Detects frequency-domain anomalies (see [Frequency Domain Analysis](#-frequency-domain-analysis)).
  - **Canny Edge Detection**: Captures unnatural edge smoothness.
  - **Local Binary Patterns (LBP)**: Identifies texture irregularities.
- **Superior Performance**: Achieves 91.6% accuracy and 0.948 AUC, outperforming baseline CNNs by 7.4% in accuracy.
- **Robustness**: Excels in challenging conditions (e.g., compression, low resolution, unseen fakes).
- **Interpretability**: Handcrafted features provide clear insights into *why* an image is fake, enhancing trust in forensic use cases.

## 📊 Results

FreqNet was evaluated on a balanced Deepfake dataset from [Kaggle](https://www.kaggle.com/datasets/manj11karki/deapraka-and-real-images), outperforming a baseline CNN across key metrics:

| Model           | Accuracy | AUC   | Precision | Recall |
|-----------------|----------|-------|-----------|--------|
| Baseline CNN     | 84.2%    | 0.881 | 0.83      | 0.81   |
| **FreqNet**     | **91.6%** | **0.948** | **0.90** | **0.93** |

**Highlights**:
- **7.4% accuracy boost** and **0.067 AUC improvement** over the baseline.
- Detects subtle artifacts (e.g., unnatural textures, edge inconsistencies) missed by CNNs.
- Reduces false positives and negatives, critical for real-world applications.

## 🔬 Frequency Domain Analysis

FreqNet leverages the **Fast Fourier Transform (FFT)** to detect frequency-domain anomalies introduced by Deepfake generation pipelines, which often leave unnatural patterns in the frequency spectrum. The 2D FFT is computed for a grayscale image \( f(x, y) \) of size \( M \times N \):

\[
F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) \cdot e^{-2 \pi i \left( \frac{ux}{M} + \frac{vy}{N} \right)}
\]

Where:
- \( f(x, y) \): Pixel intensity at spatial coordinates \( (x, y) \).
- \( F(u, v) \): Complex frequency content at coordinates \( (u, v) \).
- \( e^{-2 \pi i \cdot (\ldots)} \): Complex exponential projecting spatial content into sinusoidal components.

The magnitude spectrum \( |F(u, v)| \) is shifted to center low frequencies, and three compact descriptors are extracted:
- **Mean Frequency Magnitude**:
  \[
  \mu = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} |F(u, v)|
  \]
  Measures average strength of frequency content.
- **Standard Deviation**:
  \[
  \sigma = \sqrt{\frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} \left( |F(u, v)| - \mu \right)^2}
  \]
  Captures variation in frequency energies.
- **Skewness**:
  \[
  \text{Skew} = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} \left( \frac{|F(u, v)| - \mu}{\sigma} \right)^3
  \]
  Indicates whether energy is concentrated in high or low frequencies.

These features reveal anomalies (e.g., unnatural high-frequency noise) that distinguish Deepfakes from real images, enhancing FreqNet’s detection capabilities.

## 🎥 Demo Video

Watch FreqNet in action! Our deployment demo showcases real-time Deepfake detection on sample images and videos.

[![FreqNet Demo](https://raw.githubusercontent.com/your-username/freqnet/main/thumbnail.jpg)](https://drive.google.com/file/d/1-gnwkmd9704hmey6Sh5fnnPhhjz8-LUU/view?usp=sharing)

*Note*: Ensure the Google Drive link is set to “Anyone with the link” for accessibility. Replace `thumbnail.jpg` with your actual thumbnail image in the repository.

## 🛠️ Implementation

- **Framework**: PyTorch (GPU-accelerated).
- **Architecture**: Hybrid model with frozen ResNet-50 (up to layer 4) and handcrafted feature extraction (FFT, Canny, LBP).
- **Training**:
  - Images: 256x256, normalized with ImageNet statistics.
  - Optimizer: Adam (LR=1e-4, weight decay=1e-5).
  - Scheduler: ReduceLROnPlateau (patience=3, factor=0.5).
  - Early Stopping: Patience=5.
  - Loss: Binary cross-entropy.
- **Dataset**: Balanced Deepfake dataset from [Kaggle](https://www.kaggle.com/datasets/manj11karki/deapraka-and-real-images).

## 🚀 Deployment

FreqNet is designed for real-world deployment:
- **Use Cases**: Social media content moderation, digital forensics, law enforcement.
- **Setup**: Deployable as a standalone Python application or integrated into web platforms via Flask/Django APIs.
- **Example**: Run `deploy.py` to process images/videos in real-time:
  ```bash
  python deploy.py --model_path <path-to-trained-model> --input <image-or-video-path>
  ```
- **Streamlit App**: Try the deployed model at [Streamlit](https://aruni20-deepfake-detection-classifier-app-qlu24w.streamlit.app/).
- **Performance**: Optimized for GPU environments; CPU fallback available for low-resource settings.

## 📈 Limitations

- **Dataset Scope**: Evaluated on a single dataset; may require cross-dataset validation.
- **Frame-Based**: Lacks temporal analysis for video Deepfakes.

## 🔮 Future Work

- **Cross-Dataset Testing**: Evaluate on FaceForensics++, DFDC, Celeb-DF for broader generalizability.
- **Video Analysis**: Add temporal features (e.g., optical flow, LSTMs) for video Deepfakes.
- **Optimization**: Use lightweight models (e.g., MobileNet) or pruning for faster inference.
- **Explainability**: Integrate Grad-CAM/SHAP for transparent predictions.
- **Adversarial Defense**: Enhance robustness against attacks targeting Deepfake detectors.

## 🏗️ Getting Started

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
   - Get the Deepfake dataset from [Kaggle](https://www.kaggle.com/datasets/manj11karki/deapraka-and-real-images).
4. **Train the Model**:
   ```bash
   python model.py --dataset_path <path-to-dataset> --model_path <path-to-save-model>
   ```
5. **Run Deployment**:
   - Use the Streamlit app: [Deepfake Detection Classifier](https://aruni20-deepfake-detection-classifier-app-qlu24w.streamlit.app/).
   - Or run locally:
     ```bash
     python deploy.py --model_path <path-to-trained-model> --input <image-or-video-path>
     ```

## 👥 Author

- **Aruni Saxena** (202418006)

## 📜 Citation

```bibtex
@article{saxena2025freqnet,
  title={Deepfake Forensics Reimagined: Learning How It Was Faked, Not Just What Is Fake},
  author={Saxena, Aruni and Sagar, Krish},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.