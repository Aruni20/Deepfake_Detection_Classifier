# FreqNet: Hybrid Deepfake Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red) ![License](https://img.shields.io/badge/License-MIT-green)

## Why Deepfakes Are a Problem

Deepfakes, made with GANs or tools like DeepFaceLab, look scarily real, fueling misinformation and fraud. Standard CNNs like ResNet-50 catch big-picture facial features but miss tiny statistical clues, struggle with compressed images, and aren’t great at explaining fakes. FreqNet fixes this by digging into how Deepfakes are crafted.

## Why CNNs Fail and Classical Features Shine

CNNs focus on high-level patterns like faces or expressions, but modern Deepfakes nail those. They miss low-level artifacts: weird frequency noise, smoothed edges from blending, or odd skin textures. FreqNet tackles this with classical features. FFT spots unnatural frequency patterns, Canny catches edge inconsistencies, and LBP flags texture quirks. This hybrid approach boosts accuracy by 7.4% and makes FreqNet robust against tricky conditions like low-res images.


## Results

We tested FreqNet on a Kaggle Deepfake dataset ([link](https://www.kaggle.com/datasets/manj11karki/deapraka-and-real-images)) against a baseline CNN:

| Model          | Accuracy | AUC   | Precision | Recall |
|----------------|----------|-------|-----------|--------|
| Baseline CNN   | 84.2%    | 0.881 | 0.83      | 0.81   |
| **FreqNet**    | **91.6%** | **0.948** | **0.90**  | **0.93** |

FreqNet’s 7.4% accuracy edge comes from catching subtle artifacts, reducing errors in real-world scenarios.

## Demo Video

See FreqNet in action! Our demo shows real-time detection:

![Demo](https://raw.githubusercontent.com/your-repo/freqnet/main/thumbnail.jpg)  
[Watch Demo](https://drive.google.com/file/d/1-gnwkmd9704hmey6Sh5fnnPhhjz8-LUU/view?usp=sharing) *(Set to “Anyone with the link”)*

## Implementation

FreqNet uses PyTorch with GPU support. We froze ResNet-50 up to layer 4 for semantic features, added FFT, Canny, and LBP for handcrafted features, and trained on 256x256 images with Adam (LR=1e-4), ReduceLROnPlateau, and binary cross-entropy loss.

## Deployment

I deployed FreqNet on a Streamlit app for easy use: [Try it here](https://aruni20-deepfake-detection-classifier-app-qlu24w.streamlit.app/). It’s great for social media, forensics, or law enforcement. Locally, run:

```bash
python deploy.py --model_path <model> --input <image-or-video>
```

## Limitations

FreqNet’s tested on one dataset, so we need to check it on others like FaceForensics++. It’s also frame-based, missing video-specific clues.

## Future Plans

We’ll test on more datasets, add video analysis with temporal features, optimize with MobileNet, and use Grad-CAM for clearer explanations.

## Get Started

Clone the repo:

```bash
git clone <your-repo-url>
cd freqnet
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/manj11karki/deapraka-and-real-images).

Train the model:

```bash
python model.py --dataset_path <dataset> --model_path <save-path>
```

Deploy via Streamlit ([link](https://aruni20-deepfake-detection-classifier-app-qlu24w.streamlit.app/)) or locally:

```bash
python deploy.py --model_path <model> --input <image-or-video>
```

## Authors

Aruni Saxena (202418006)  
Krish Sagar (202418020)

## Citation

```bibtex
@article{saxena2025freqnet,
  title="Deepfake Forensics: Learning How It Was Faked, Not Just What Is Fake",
  author={Saxena, Aruni and Sagar, Krish},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE).
