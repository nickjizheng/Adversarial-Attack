# Adversarial Attack Experiments

This repository contains research code for evaluating how adversarial perturbations to video activity recognition can affect downstream assistant responses in safety-oriented settings.

This repository accompanies my published research paper: [Adversarial Attack Research Paper](docs/adversarial-attack-research-paper.pdf).

DOI: [https://doi.org/10.61173/0r0nva44](https://doi.org/10.61173/0r0nva44)

The experiments use UCF101-style activity videos, a fine-tuned ResNet18 frame classifier, FGSM perturbations, and GPT-based assistant prompts. Response shifts are compared with semantic and syntactic similarity metrics.

## Repository Layout

```text
.
|-- src/
|   |-- fine_tune.py              # Fine-tune ResNet18 on video frames
|   |-- main_uni.py               # Label-mediated/unimodal assistant pipeline
|   |-- main_multi.py             # Frame-prompted multimodal assistant pipeline
|   |-- mix_baby_crawling.py      # Baby safety comparison script
|   `-- mix_biking.py             # Road safety comparison script
|-- docs/
|   |-- adversarial-attack-research-paper.pdf
|   `-- research-rationale.md     # Research motivation and current caveats
|-- checkpoints/                  # Local model weights, ignored by git
|-- outputs/                      # Generated plots/logs, ignored by git
|-- .env.example                  # Environment variable template
|-- .gitignore
`-- requirements.txt
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the values for your machine:

```bash
copy .env.example .env
```

Place validation videos under `data/validation` using one folder per class, or point `VALIDATION_DATA_DIR` to an existing validation set. Place the fine-tuned checkpoint at `checkpoints/fine_tuned_resnet18.pth`, or set `MODEL_CHECKPOINT`.

Model weights and datasets are intentionally ignored by git.

## Running Experiments

Run the label-mediated pipeline:

```bash
python src/main_uni.py
```

Run the frame-prompted pipeline:

```bash
python src/main_multi.py
```

Run the combined case studies:

```bash
python src/mix_baby_crawling.py
python src/mix_biking.py
```

Fine-tune the ResNet18 classifier:

```bash
python src/fine_tune.py
```

## Notes Before Publishing

Add a license before publishing publicly. If the trained checkpoints need to be shared, use GitHub Releases, Hugging Face, Google Drive, or another artifact store rather than committing them directly.
