# Listen, Chat, and Remix (LCR)

[![arXiv](https://img.shields.io/badge/arXiv-2402.03710-b31b1b.svg)](https://arxiv.org/abs/2402.03710)
[![Website](https://img.shields.io/badge/Website-Demo-blue)](https://listenchatremix.github.io/demo)

Official implementation of **"Listen, Chat, and Remix: Text-Guided Soundscape Remixing for Enhanced Auditory Experience"**.

---

## Overview

**Listen, Chat, and Remix (LCR)** is a text-guided sound enhancement system capable of arbitrarily remixing every source in speech and audio mixtures. Unlike traditional models that isolate a single source, LCR can simultaneously extract, remove, or adjust the volumes of multiple sounds in a single step based on open-vocabulary natural-language instructions.

<img width="652" height="305" alt="Screenshot 2026-02-04 at 12 54 34 AM" src="https://github.com/user-attachments/assets/ff76b600-74c4-48ca-97b2-fc23c12a9600" />

---

## System Architecture

The LCR system consists of two primary components jointly optimized in an end-to-end manner:

1.  **PromptReader:** A language model (GPT-2 or LLaMA2 with LoRA) that generates a D-dimensional **semantic filter** from the text prompt.
2.  **SoundRemixer:** An acoustic model that applies the semantic filter to estimate a **remixing mask**, which selectively filters the latent representation of the input mixture.

<img width="883" height="433" alt="Screenshot 2026-02-04 at 12 55 48 AM" src="https://github.com/user-attachments/assets/28e35a76-b1ab-4451-b9f0-c28122dc4352" />

---

## Repository Structure

```text
├── data/datasets          # Scripts for handling sound sources and mixtures
├── hparams                # Hyperparameter configurations for different models
├── modules                # Core implementations of SoundRemixer and PromptReader
├── samples                # Audio samples and demo mixtures
├── save/                  # Checkpoints
├── utils                  # Helper functions for audio processing and evaluation
├── finetune_LCR.py        # Main script for training (finetuning)
├── prepare_data.ipynb     # Notebook for dataset generation and preprocessing
├── run_LCR.ipynb          # Notebook for inference and testing your own audio
└── requirements.txt       # Environment dependencies
```
---

## Inference
In terminal
```text
pip install -r requirements.txt
```
Then, follow run_LCR.ipynb,
which does
1. **Initialize Models**: Setup the ***PromptReader*** and ***SoundRemixer*** using the `modules/` and `hparams/`.
2. **Load Checkpoint**: Load pre-trained weights (e.g., from `save/pretrain_sepformer_llama2_lora`).
3. **Input Audio & Text**: Provide a 16 kHz sound mixture and a text prompt.
4. **Estimate Semantic Filter**: The ***PromptReader*** interprets the text to encode target sources and actions
5. **Apply Remix Mask**: The ***SoundRemixer*** estimates a mask to scale or remove components in the latent space.
6. **Decode Output**: The decoder maps the filtered latent back to the remixed waveform.

---

## Checkpoints and Test Data

We provide the Sepformer + LLaMA 2 checkpoint [here](https://drive.google.com/drive/folders/1jHlMbrXmaxkhdfn5QtFRGhjOUgdxEoFj?usp=sharing), along with 2 Speech + 2 Audio mixtures and text prompts for the Target Speech Extraction, Target Audio Extraction, and Multiple Speech/Audio Extraction tasks.

## Citation
```text
@article{jiang2025listen,
  title={Listen, Chat, and Remix: Text-Guided Soundscape Remixing for Enhanced Auditory Experience},
  author={Jiang, Xilin and Han, Cong and Li, Yinghao Aaron and Mesgarani, Nima},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  year={2025},
  publisher={IEEE}
}
```
