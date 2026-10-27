
import sys
import os
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Allow importing from src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from inference import predict_image, generate_gradcam


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Skin Disease Detection AI",
    page_icon="🔬",
    layout="centered"
)


# -----------------------------
# Header
# -----------------------------

st.title("🔬 Skin Disease Detection AI")

st.write(
    "Upload a dermoscopic skin-lesion image to receive "
    "AI-generated predictions."
)

st.warning(
    "Educational / research tool only. "
    "This system is not a medical diagnosis and should not "
    "replace evaluation by a qualified healthcare professional."
)


# -----------------------------
# Upload image
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a dermoscopic image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Save uploaded image temporarily
    temp_dir = os.path.join(PROJECT_ROOT, "app", "temp")
    os.makedirs(temp_dir, exist_ok=True)

    image_path = os.path.join(
        temp_dir,
        uploaded_file.name
    )

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display image
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width="stretch"
    )

    # Run prediction
    if st.button("Analyze Image"):

        with st.spinner("Analyzing image..."):

            result = predict_image(
                image_path,
                top_k=3
            )

        # -----------------------------
        # Image quality
        # -----------------------------

        st.subheader("Image Quality")

        quality = result["quality"]

        if quality["quality_ok"]:
            st.success("Image quality check passed.")
        else:
            st.error("Image quality check failed.")

        if quality["warnings"]:
            for warning in quality["warnings"]:
                st.write("⚠️", warning)

        # -----------------------------
        # Predictions
        # -----------------------------

        st.subheader("Top-3 Predictions")

        for i, prediction in enumerate(
            result["predictions"],
            start=1
        ):

            confidence = prediction["confidence"] * 100

            st.write(
                f"**{i}. {prediction['class']}**"
            )

            st.progress(
                min(int(confidence), 100)
            )

            st.write(
                f"Confidence: **{confidence:.2f}%**"
            )

            st.divider()



        # -----------------------------
        # Explainable AI - Grad-CAM
        # -----------------------------

        st.subheader("🔎 Explainable AI")

        st.write(
            "Grad-CAM highlights the image regions that "
            "influenced the model's prediction."
        )

        try:
            original_image, heatmap, predicted_name = generate_gradcam(
                image_path
            )

            heatmap_resized = tf.image.resize(
                heatmap[..., np.newaxis],
                (original_image.shape[0], original_image.shape[1])
            ).numpy().squeeze()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(heatmap_resized, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap")
            axes[1].axis("off")

            axes[2].imshow(original_image)
            axes[2].imshow(
                heatmap_resized,
                cmap="jet",
                alpha=0.45
            )
            axes[2].set_title("Prediction Explanation")
            axes[2].axis("off")

            plt.tight_layout()

            st.pyplot(fig)

            st.caption(
                "The highlighted regions indicate areas that "
                "influenced the model's prediction. They are "
                "not medical evidence or a diagnostic finding."
            )

        except Exception as e:
            st.error(f"Grad-CAM could not be generated: {e}")

        st.info(
            "Predictions are model outputs and may be incorrect. "
            "Do not use them for self-diagnosis or treatment decisions."
        )
