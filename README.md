# 🐾 Cat vs Dog Classifier for Shelters – FastAI + Gradio + Hugging Face

This repository contains a deep learning project built with **FastAI** to classify images as either **cats** or **dogs**, fine-tuned with transfer learning and deployed via a **Gradio interface** on **Hugging Face Spaces**.

## 🧠 Project Overview

This project was developed as a part of a deep learning assignment. The goal was to:

- Build a cat vs dog classifier using the **Oxford-IIIT Pets dataset**
- Apply advanced fine-tuning techniques including:
  - Learning Rate Finder
  - Discriminative Learning Rates
  - Freezing & Unfreezing
- Set up a **confidence threshold** (95%) to return **“Unknown”** for uncertain predictions
- Deploy the final model as an interactive app accessible through the browser

> 📌 This can be a practical use-case for **animal shelters**, assisting in organizing or pre-classifying pet images.

---

## 🚀 Demo

🔗 **Live App**: [Try it on Hugging Face Spaces]((https://huggingface.co/spaces/semihozsoy/cat-vs-dog))

📸 Upload a cat or dog image. If the model is more than 95% confident, it will classify it; otherwise, it will return "Unknown or Uncertain".

---

## 🗂 Project Structure

```
.
├── app.py               # Gradio interface
├── midterm.ipynb        # Full model training pipeline (Google Colab)
├── export.pkl           # Trained FastAI model
├── requirements.txt     # Dependencies for Gradio deployment
└── README.md            # You're here!
```

---

## ⚙️ How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/smhzsy/ada447.git
cd cat-dog-classifier
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python app.py
```

Then open your browser at `http://localhost:7860/`.

---

## 📊 Model Accuracy

- **Validation Accuracy:** 99.79%
- **Confidence Threshold:** 95%
- **Behavior for Out-of-Domain Inputs:** Returns "Unknown" if input doesn't resemble a cat or dog (e.g., a car)

---

## ✍️ Author

**Semih Özsoy**  
Cybersecurity enthusiast, AI researcher for fun

📬 For questions, feel free to reach out or open an issue.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
