# 🍔 Food Classification with Vision Transformers (ViT)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-omsingh4459-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/om-singh-5b8032222/)
[![GitHub](https://img.shields.io/badge/GitHub-omsingh4459-green?style=for-the-badge&logo=github)](https://github.com/omsingh4459)

A deep learning project that leverages the power of Vision Transformers (ViT) to classify 101 different types of food items. This repository contains the complete workflow, from data preprocessing to model training and evaluation, implemented in PyTorch.

### ✨ Project Demo

*(I highly recommend you create a short GIF or add a few screenshots showing the model classifying different food images. This is extremely effective for recruiters.)*

**[Insert a GIF or a few screenshots here showing your model's predictions]**

*Example: An image of pizza being correctly classified as "Pizza."*

---

### 🎯 Project Overview

This project tackles the challenge of fine-grained image classification by building a robust model to identify various food dishes. The core of this project is the implementation of a Vision Transformer, a state-of-the-art architecture that applies the transformer model, originally designed for NLP, to computer vision tasks. The model was trained on the popular **Food-101 dataset**.

### 🚀 Key Features

- **Modern Architecture:** Implements the **Vision Transformer (ViT)** model, showcasing skills with cutting-edge deep learning techniques.
- **End-to-End Workflow:** Includes complete data preprocessing, augmentation, model training, and evaluation scripts.
- **High Accuracy:** The trained model achieves a strong accuracy in classifying 101 distinct food categories.
- **Tech Stack:** Built with **Python**, **PyTorch**, **TorchVision**, **Matplotlib**, and **NumPy**.

---

### 🔧 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

#### Prerequisites

- Python 3.8+
- PyTorch
- TorchVision
- Matplotlib
- NumPy

#### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/omsingh4459/Food-Classification-using-Vision-Transformer.git](https://github.com/omsingh4459/Food-Classification-using-Vision-Transformer.git)
    cd Food-Classification-using-Vision-Transformer
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your local environment.)*

---

### ⚙️ Usage

The main logic is contained within the `Food_Classifier_ViT.ipynb` Jupyter Notebook.

1.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Open `Food_Classifier_ViT.ipynb`.
3.  You can run the cells sequentially to perform:
    - Data loading and preprocessing.
    - Model definition and training.
    - Evaluation and visualization of results.

---

### 🤖 Model Architecture

This project utilizes a pre-trained Vision Transformer (`ViT-B/16`) from the `torchvision` library. The model's architecture consists of:

1.  **Image Patching:** The input image is split into fixed-size patches.
2.  **Patch & Position Embedding:** Each patch is linearly embedded and combined with position embeddings to retain spatial information.
3.  **Transformer Encoder:** The sequence of embedded patches is processed by a standard Transformer encoder, which uses multi-head self-attention to capture relationships between patches.
4.  **Classification Head:** The output from the encoder is passed to a final classification layer (`MLP Head`) to predict the food category.

The final classifier head was customized for the Food-101 dataset, which has 101 output classes.

---

### 📈 Results & Performance

The model was trained for **[Enter Number of Epochs, e.g., 10]** epochs, achieving the following performance:

- **Training Accuracy:** `[Enter your final training accuracy, e.g., ~85%]`
- **Validation Accuracy:** `[Enter your final validation accuracy, e.g., ~80%]`

Here is a look at the training and validation loss curves over the epochs:

**[Insert a screenshot of your training/validation loss graph from the notebook]**

---

### 💡 Future Improvements

- **Hyperparameter Tuning:** Systematically tune hyperparameters like learning rate, optimizer, and batch size to further boost accuracy.
- **Deploy the Model:** Deploy the trained model as a web application using Flask or FastAPI to allow real-time predictions.
- **Experiment with Other Architectures:** Compare the performance of ViT with other modern CNN architectures like EfficientNetV2.

---

### Let's Connect!

I'm passionate about building impactful deep learning solutions. If you have any questions or would like to collaborate, feel free to reach out!

- **LinkedIn:** [https://www.linkedin.com/in/om-singh-5b8032222/](https://www.linkedin.com/in/om-singh-5b8032222/)
- **Email:** `[Your Email Address]`