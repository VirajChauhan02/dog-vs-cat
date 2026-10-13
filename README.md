# Cat and Dog Image Classifier

This project trains a convolutional neural network (CNN) on labelled cat and dog images and serves predictions through a Streamlit web app. It accepts an uploaded image and predicts **Cat** or **Dog**.

The supplied project ZIP contained a notebook and a Tkinter example, but no dataset or trained model. Its notebook also uses a computer-specific path. This deployment package uses portable paths and a web interface. It does not claim an accuracy score; evaluate the trained model before reporting one.

## Requirements

- Python 3.11
- Windows PowerShell for the one-command local setup
- The public [Kaggle Cat and Dog dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog), licensed CC0

## Dataset access

This package uses a public Kaggle dataset mirror rather than the original Dogs vs. Cats competition, which is closed to new entrants. The mirror contains labelled cat and dog folders and is licensed CC0. KaggleHub downloads it and copies readable images into `data/train/` with the labels expected by the trainer. If Kaggle asks for authentication, use a local Kaggle token; never share it or commit it to GitHub.

## Run locally

Open PowerShell in this project folder and run:

```powershell
.\start.ps1
```

On the first run, this creates a local Python environment and installs the pinned packages. If `model.keras` does not exist, it downloads the dataset (if needed), trains the CNN from `data/train`, and saves the model. Then it starts the app. Later runs reuse the environment and model, so download and training do not repeat unless you remove the data or model files.

The model uses 128 x 128 RGB images, scales pixel values to 0 to 1, and reserves a stratified 20% validation split from labelled images. Data augmentation is applied only during training. The Kaggle competition's unlabelled test images are not downloaded or used to calculate accuracy.

## Deploy on Streamlit Community Cloud

Streamlit Community Cloud deploys an app from a GitHub repository. First run the local setup above so `model.keras` has been created. Keep the training dataset out of GitHub; `data/` is excluded by `.gitignore`.

1. Create a GitHub repository under an account you control and upload this deployment folder, including `app.py`, `requirements.txt`, and `model.keras`.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and grant it access to that repository.
3. Create an app using the repository, branch, and `app.py` as the entry point. In **Advanced settings**, select Python 3.11.
4. Deploy the app. Streamlit can update it when you commit later changes to the connected GitHub repository.

Do not upload Kaggle credentials, API tokens, or the full training dataset to a public repository. The trained model is required by the deployed app. If it is too large for the selected GitHub workflow, store it in a model artifact service and configure the app to load it securely.

## Files

- `app.py` - Streamlit upload and prediction interface.
- `download_data.py` - KaggleHub download and preparation of labelled images.
- `train_model.py` - CNN training, validation split, and model export.
- `start.ps1` - One-command local setup, first-time training, and app launch.
- `requirements.txt` - Pinned Python dependencies.

