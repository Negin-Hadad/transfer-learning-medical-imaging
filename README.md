# Multi-Label Retinal Disease Classification via Deep Learning

Retinal Disease Classification via Deep Learning
This repository contains a multi-label deep learning pipeline for detecting Diabetic Retinopathy (DR), Glaucoma (G), and Age-related Macular Degeneration (AMD) in retinal images. Developed for the University of Oulu Deep Learning course (5211535-3006-2025), the project utilizes transfer learning and ensemble strategies to achieve robust diagnostic performance on the ODIR dataset.

## Project Overview
This project implements a deep learning pipeline for the multi-label classification of retinal diseases: **Diabetic Retinopathy (DR)**, **Glaucoma**, and **Age-related Macular Degeneration (AMD)**. It leverages transfer learning, custom loss functions for class imbalance, attention mechanisms, and ensemble strategies to optimize performance on the ODIR dataset. The project consists of four tasks:

**Task 1: Transfer Learning Baselines** - Evaluation of ResNet18 and EfficientNet backbones using strategies like Frozen Backbone and Full Fine-Tuning.
**Task 2: Handling Class Imbalance** - Implementation of Focal Loss and Class-Balanced Loss to improve detection of minority classes (Glaucoma, AMD).
**Task 3: Attention Mechanisms** - Integration of Squeeze-and-Excitation (SE) Blocks and Multi-Head Attention (MHA) to enhance feature extraction.
**Task 4: Ensemble Learning** - Implemented weighted averaging, max voting, stacking with meta-learners, and adaptive threshold optimization to integrate diverse models from Tasks 1, 2, and 3 for improved predictive performance.

---

## File Structure
DLsns.zip  
│
├── DLsns.py # Main source code (Tasks 1–4)  
├── README.md # This file  
│  
├── models/  
│ ├── DLsns_task1-1.pt # Task 1 – No Fine-tuning  
│ ├── DLsns_task1-2.pt # Task 1 – Frozen Backbone  
│ ├── DLsns_task1-3.pt # Task 1 – Full Fine-tuning  
│ ├── DLsns_task2-1.pt # Task 2 – Focal Loss  
│ ├── DLsns_task2-2.pt # Task 2 – Class-Balanced Loss  
│ ├── DLsns_task3-se.pt # Task 3 – SE Attention  
│ └── DLsns_task3-mha.pt # Task 3 – MHA Attention 

---

## Requirements
The code was developed and tested using:

- Python 3.9+
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- Pillow
- Google Colab (recommended for GPU support)

## 🚀 Usage Instructions

The project logic is contained in `dlsns.py`.

Install dependencies using:
```bash
pip install torch torchvision numpy pandas scikit-learn pillow