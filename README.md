FreqNet: Hybrid Deepfake Detection Framework

Introduction

Deepfakes—synthetically generated media that mimic real human appearances—pose a significant threat to digital media authenticity, enabling misinformation, biometric spoofing, and evidence tampering. Traditional Convolutional Neural Network (CNN)-based detectors, such as ResNet-50, XceptionNet, and EfficientNet, excel at capturing high-level semantic features (e.g., facial identity, pose) but often miss subtle statistical artifacts introduced by generative models like GANs or autoencoders. These artifacts, such as unnatural frequency patterns, edge discontinuities, or texture irregularities, are critical for distinguishing high-quality Deepfakes from authentic media.

This project introduces FreqNet, a novel hybrid Deepfake detection model that combines the semantic power of deep learning with classical image processing techniques. By integrating ResNet-50’s deep features with handcrafted descriptors—Fast Fourier Transform (FFT) for frequency analysis, Canny edge detection for edge structure, and Local Binary Patterns (LBP) for texture irregularities—FreqNet captures both what an image portrays and how it was constructed. This dual-branch approach achieves superior accuracy and robustness, making it a promising tool for digital forensics, social media moderation, and law enforcement.

Problem Statement

Deepfakes, powered by advances in Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and face-swapping frameworks like DeepFaceLab, produce highly realistic media that challenge public trust. While CNN-based detectors achieve high accuracy under controlled conditions (e.g., >90% on datasets like FaceForensics++ or Celeb-DF), they struggle with:





Subtle Artifacts: Modern Deepfakes preserve semantic correctness but leave statistical inconsistencies (e.g., abnormal high-frequency noise, unnatural edge smoothness).



Generalization: Performance degrades on unseen manipulation types, compressed media, or low-resolution inputs.



Interpretability: Pure CNN models lack transparency in identifying why an image is fake.

FreqNet addresses these challenges by focusing on how Deepfakes are made, not just what they depict, through a fusion of deep and handcrafted features.

Novelty of FreqNet

FreqNet’s key innovation lies in its hybrid architecture, which bridges deep learning and classical image processing to detect both semantic and statistical cues. Unlike traditional CNNs, FreqNet:





Fuses Deep and Handcrafted Features:





Deep Features: Leverages a pre-trained ResNet-50 backbone to extract high-level semantic representations (e.g., facial identity, expression).



Handcrafted Features:





FFT: Captures frequency-domain inconsistencies (mean, standard deviation, skewness) to detect unnatural patterns left by generative models.



Canny Edge Detection: Quantifies edge sharpness/smoothness to identify blending artifacts.



LBP: Encodes texture irregularities to flag synthetic skin or eye regions.



These features are concatenated into a 2080-dimensional vector, enabling a holistic analysis of Deepfake artifacts.



Improved Robustness: Outperforms CNN-only models under challenging conditions like compression, low resolution, or unseen manipulation types.



Enhanced Interpretability: Handcrafted features provide insights into why an image is classified as fake, supporting forensic applications.



Real-World Applicability: Designed for deployment in digital media forensics, social media content moderation, and law enforcement.

Dataset

FreqNet was trained and evaluated on a balanced Deepfake dataset curated by Karki et al. (available on Kaggle). The dataset includes:





Real Images: Diverse photographs and video frames with varied lighting, angles, and expressions.



Fake Images: Generated using multiple synthesis pipelines (e.g., GANs, autoencoders, encoder-decoder architectures), replicating real-world forgery techniques.

Results

FreqNet was evaluated against a baseline CNN using standard metrics: Accuracy, Area Under the ROC Curve (AUC), Precision, and Recall. The results demonstrate significant improvements:







Model



Accuracy (%)



AUC



Precision



Recall





Baseline CNN



84.2



0.881



0.83



0.81





FreqNet



91.6



0.948



0.90



0.93

Key Observations:





Higher Accuracy and AUC: FreqNet achieves 91.6% accuracy and 0.948 AUC, outperforming the baseline by 7.4% in accuracy and 0.067 in AUC.



Improved Precision and Recall: Reduces false positives (higher precision) and misses fewer Deepfakes (higher recall).



Qualitative Insights: Excels at detecting subtle artifacts (e.g., unnatural textures, edge inconsistencies) missed by CNNs, though challenges remain with low-quality real images or highly realistic fakes.

Implementation





Framework: PyTorch with GPU acceleration.



Architecture: Hybrid model with a frozen ResNet-50 backbone (up to layer 4) and handcrafted feature extraction (FFT, Canny, LBP).



Training:





Images resized to 256x256 and normalized using ImageNet statistics.



Adam optimizer (learning rate: 1e-4, weight decay: 1e-5).



ReduceLROnPlateau scheduler (patience=3, factor=0.5).



Early stopping (patience=5).



Binary cross-entropy loss.



Hardware: Trained with GPU support for efficient processing.

Limitations





Dataset Dependency: Evaluated on a single dataset, which may not capture all real-world Deepfake variations.



Computational Overhead: Handcrafted feature extraction increases complexity, potentially limiting real-time performance.



Frame-Based Analysis: Misses temporal inconsistencies in Deepfake videos.

Future Work





Cross-Dataset Evaluation: Test on diverse datasets (e.g., FaceForensics++, DFDC, Celeb-DF) for broader generalizability.



Temporal Analysis: Incorporate motion-based features or recurrent architectures (e.g., LSTMs) for video Deepfakes.



Model Optimization: Explore lightweight architectures or pruning for real-time deployment.



Explainability: Integrate Grad-CAM or SHAP for transparent predictions.



Adversarial Robustness: Enhance resilience against adversarial attacks targeting Deepfake detectors.
