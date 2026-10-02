# MV Project — Classical Features and Learned Representations for Image Recognition

This repository contains two Jupyter notebooks for a machine-vision course project. Together, they compare two approaches to image recognition:

1. **Hand-crafted visual descriptors** (shape, colour, and texture) paired with traditional machine-learning classifiers on CIFAR-100.
2. **Learned compact representations** from autoencoders and PCA, paired with nearest-neighbour and neural-network classifiers for face recognition on a sampled CASIA-WebFace subset.

The notebooks are written and executed as Kaggle notebooks. They retain their recorded outputs, including feature dimensions, training progress, and experiment results, so the repository serves as both an implementation and an experiment record.

## Repository contents

| File | Purpose |
| --- | --- |
| [`mv-project.ipynb`](mv-project.ipynb) | CIFAR-100 experiment: extract HOG, HSV colour-histogram, and LBP descriptors; then evaluate KNN, K-Means, and linear SVM classifiers. |
| [`mv-milestone2_final_ISA.ipynb`](mv-milestone2_final_ISA.ipynb) | Face-recognition experiment: learn 64-dimensional image embeddings with vanilla, variational, and convolutional autoencoders; compare them with 64-component PCA (Eigenfaces). |
| `README.md` | Project overview, experiment design, recorded results, and instructions for reproducing the notebooks. |

## Part 1: CIFAR-100 feature-engineering study

### Goal

Measure how well three classical image descriptors support 100-class object recognition when used with simple supervised and unsupervised methods.

### Data

The notebook loads CIFAR-100 with `tensorflow.keras.datasets.cifar100`:

| Split | Images | Image shape | Classes |
| --- | ---: | --- | ---: |
| Training | 50,000 | 32 × 32 RGB | 100 |
| Test | 10,000 | 32 × 32 RGB | 100 |

Labels are flattened from `(n, 1)` to `(n,)`. The notebook also normalizes pixel values to `[0, 1]`; feature extraction itself is performed on the loaded RGB images. A generated `cifar100_classes.py` helper provides readable class names for the visual examples.

### Descriptors

Each descriptor is extracted for every training and test image, then saved together in a pickle file for reuse.

| Descriptor | Method | Settings in notebook | Vector size |
| --- | --- | --- | ---: |
| HOG | Histogram of Oriented Gradients | RGB → grayscale; `pixels_per_cell=(8, 8)`; `cells_per_block=(2, 2)` | 324 |
| HSV histogram | Global colour distribution | RGB → HSV; 32 bins for each of hue, saturation, and value; each channel histogram is normalized | 96 |
| LBP | Local Binary Pattern texture histogram | RGB → grayscale; uniform LBP with radius 1 and 8 sampling points | 10 |

The notebook visualizes a HOG image, grayscale image, rescaled HOG image, the three HSV channel histograms, and an LBP histogram before processing the complete data set. Its recorded full extraction time is about 254 seconds in the Kaggle environment.

### Models and evaluation

For each descriptor independently, the notebook runs:

- **KNN** — a custom, brute-force Euclidean-distance classifier using `k=3` and majority voting.
- **K-Means** — a custom implementation with 100 randomly initialized centroids and up to 100 iterations. Cluster IDs are mapped to their dominant ground-truth class only for evaluation; this mapping uses the training labels, so the reported K-Means results are an in-sample clustering-purity-style measurement, not a held-out clustering score.
- **Linear SVM** — `sklearn.svm.SVC(kernel='linear')` trained on the complete training descriptor set.

The supervised models are evaluated on the CIFAR-100 test split with accuracy, weighted precision, weighted recall, weighted F1, confusion matrices, and (where included) classification reports. Feature vectors, prediction arrays, cluster labels, and SVM models can be saved and reloaded through pickle, NumPy `.npy`, and Joblib files.

### Recorded results

The following values are the rounded metrics printed by the notebook’s saved execution. They are useful as a baseline, but will vary with the random K-Means initialization and environment.

| Feature set | KNN test accuracy | Linear SVM test accuracy | K-Means training accuracy* |
| --- | ---: | ---: | ---: |
| HOG | 0.20 | 0.21 | 0.11 |
| HSV colour histogram | 0.14 | 0.11 | 0.08 |
| LBP | 0.04 | 0.06 | 0.06 |

\* K-Means accuracy is calculated after assigning each cluster its majority training label; see the evaluation note above.

In this experiment, gradient-based shape information (HOG) is the strongest of the three individual descriptors. The results also illustrate the limitation of using a single global descriptor on CIFAR-100’s fine-grained, 100-class recognition task.

## Part 2: face embeddings and recognition study

### Goal

Compare compact 64-dimensional image representations learned by autoencoders with a PCA/Eigenfaces baseline, then test whether those representations preserve identity information.

### Data preparation

The notebook samples the CASIA-WebFace directory provided in Kaggle as `/kaggle/input/casia-webface/casia-webface`.

- It selects **75 identity folders** that contain at least 200 images.
- It randomly samples **200 RGB images per identity**.
- Every image is resized to **128 × 128** pixels.
- The resulting data set contains **15,000 images** with labels `0–74`.
- Images are shuffled, then normalized from `[0, 255]` to `[0, 1]`.
- A two-stage `train_test_split` creates a 70% / 15% / 15% split:
  - training: 10,500 images
  - validation: 2,250 images
  - test: 2,250 images

Because class and image selection use Python and NumPy random sampling without a fixed seed, a new run can choose a different identity subset and image sample.

### Representation models

All methods produce a 64-value representation per image, allowing a like-for-like comparison.

| Representation | Encoder | Decoder / reconstruction path | Training setup |
| --- | --- | --- | --- |
| Vanilla autoencoder | Flattened 128 × 128 × 3 input → Dense 512 → 256 → 128 → 64 | Dense 64 → 128 → 256 → 512 → 49,152, then reshape to RGB image | Nadam, MSE, 20 epochs, batch size 50 |
| Variational autoencoder (VAE) | Flattened input → Dense 512 → 256 → 128 → `z_mean` and `z_log_var` (64 each) → reparameterized sample | Dense 64 → 128 → 256 → 512 → 49,152, then reshape | Adam, reconstruction MSE plus the model’s KL-divergence loss, 20 epochs, batch size 50 |
| Convolutional autoencoder | Conv2D 32 → pool → Conv2D 64 → pool → Conv2D 128 → pool → flatten → Dense 64 | Dense to 16 × 16 × 128 → transposed convolutions and upsampling back to 128 × 128 × 3 | Adam, MSE, 20 epochs, batch size 50 |
| PCA / Eigenfaces | PCA fitted to flattened training images | Inverse transform used for a reconstruction illustration | 64 components, whitening enabled, `random_state=42` |

The notebook saves the three trained autoencoders as HDF5 (`.h5`) files. It also plots reconstruction training/validation histories and visualizes the first 15 principal components plus a PCA reconstruction example.

### Recognition evaluation

The project extracts 64-dimensional vectors for each training and test image:

- vanilla autoencoder bottleneck;
- VAE mean vector (`z_mean`);
- convolutional autoencoder bottleneck; and
- PCA projection.

It evaluates two classification approaches:

1. **Euclidean nearest neighbour** — each test embedding receives the label of its closest training embedding.
2. **Dense neural classifier** — a shared architecture of Dense 128 (ReLU) → Dense 64 (ReLU) → Dense 75 (softmax), trained for 100 epochs with Adam and sparse categorical cross-entropy. The same model instance is recompiled and trained sequentially for each embedding type.

### Recorded recognition accuracy

| Representation | Euclidean nearest-neighbour accuracy | Dense classifier test accuracy |
| --- | ---: | ---: |
| Vanilla autoencoder | 24.98% | 0.27 |
| Variational autoencoder (`z_mean`) | 13.64% | 0.19 |
| Convolutional autoencoder | 26.18% | 0.26 |
| PCA / Eigenfaces | **32.89%** | 0.21 |

The Euclidean results show the PCA/Eigenfaces representation as the strongest recorded nearest-neighbour baseline, followed by the convolutional autoencoder. The dense-classifier runs have noticeably higher training than test accuracy in the stored output, a sign that these small classification heads overfit the sampled split; validation monitoring or early stopping would be a sensible next refinement.

## Running the notebooks

### Recommended: Kaggle

The notebook paths and precomputed artifact loads are configured for Kaggle. Attach the required input data sets to a Kaggle notebook:

- `mv-project.ipynb`: CIFAR-100 downloads automatically through Keras; its optional reload cells expect Kaggle inputs containing `features.pkl`, saved predictions, and SVM Joblib files.
- `mv-milestone2_final_ISA.ipynb`: attach CASIA-WebFace so the directory is available at `/kaggle/input/casia-webface/casia-webface`.

The first cell in `mv-project.ipynb` creates its helper class-name module at `/kaggle/working/cifar100_classes.py`. Run the notebook top to bottom when recreating features or models. Reload cells are optional shortcuts for previously generated artifacts; skip them when you are training from scratch.

### Local environment

For a local run, use Python 3.10+ and install the notebook dependencies:

```bash
python -m pip install tensorflow numpy matplotlib scikit-image opencv-python scikit-learn scipy tqdm pillow joblib jupyter
jupyter notebook
```

Before execution, replace the hard-coded `/kaggle/input/...` and `/kaggle/working/...` paths with local paths. In the face notebook, set `base_dir` to the root directory containing the identity folders. The data sets and generated artifacts are intentionally not committed to this repository because of their size.

## Notes and limitations

- This is an experimental notebook repository, not a packaged application. There is no command-line entry point or pinned dependency lock file.
- Several long-running stages use brute-force distance calculations, full-data SVM fitting, or dense image reconstruction. A GPU and sufficient RAM are recommended, particularly for the face experiment.
- The first notebook’s custom K-Means can produce empty clusters, which would yield undefined centroid values; repeated runs can also differ because centroids are randomly initialized.
- CIFAR-100 feature results compare descriptors one at a time. Combining and scaling descriptors, using a faster KNN implementation, and tuning SVM hyperparameters are natural extensions.
- The face notebook uses random sampling and does not stratify the train/validation/test split. Set explicit random seeds and use stratification for more reproducible comparisons.

## Original Kaggle notebook

The CIFAR-100 notebook includes an **Open in Kaggle** badge linking to its original Kaggle version: [MV Project on Kaggle](https://www.kaggle.com/code/ammarbkhet/mv-project?scriptVersionId=212687060).
