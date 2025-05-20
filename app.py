from fastai.vision.all import *
import gradio as gr
from pathlib import Path

def label_func(f): return Path(f).name[0].isupper()

learn = load_learner("export.pkl")

def classify_image(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)

    confidence = float(probs[pred_idx])
    labels = learn.dls.vocab

    if confidence < 0.95:
        return {"🐾 Unknown or Uncertain": 1.0}

    label_map = {
        "0": "🐶 Dog",
        "1": "🐱 Cat"
    }

    return {label_map.get(str(i), labels[i]): float(probs[i]) for i in range(len(probs))}

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload an Image 🖼️"),
    outputs=gr.Label(num_top_classes=2, label="Prediction Result 🎯"),
    title="🐶 Cat vs Dog Classifier 🐱",
    description="Upload an image of a cat or a dog and let the model guess it!\n\n⚠️ If the model is not sure (confidence < 95%), you'll see an 'Unknown or Uncertain' label.",
    theme="soft",
    allow_flagging="never"
)

demo.launch()
