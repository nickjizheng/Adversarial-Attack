# Research Rationale

The core question behind this project is whether adversarial changes to video perception can alter higher-level assistant judgments in safety-relevant contexts.

The experiments chain together three stages:

1. Sample frames from UCF101-style videos.
2. Classify activity using a fine-tuned ResNet18 model, optionally after an FGSM-style perturbation.
3. Ask an assistant to make a safety judgment from either the predicted activity label or selected video frames.

This matters because many deployed AI systems are pipelines rather than isolated classifiers. In those systems, the most important failure is not only whether a classifier label changes. The more consequential question is whether a small input perturbation changes the final decision, such as whether a baby might need help or whether a cyclist is behaving safely.

The repository currently explores two variants:

- **Unimodal / label-mediated:** the assistant receives the activity label produced by the vision model.
- **Multimodal / frame-prompted:** the assistant receives selected frame images as part of the prompt.

The output responses are compared with semantic and syntactic similarity measures using BERT and SentenceTransformer embeddings.

## Current Caveat

The multimodal scripts should be reviewed before using them as evidence of a direct visual attack against the assistant. In the current implementation, adversarial tensors are used for classifier inference, but the base64 images sent to the assistant appear to be encoded from the original frame objects. That means the code may currently demonstrate an attacked intermediate classifier more strongly than an attacked end-to-end multimodal input.

