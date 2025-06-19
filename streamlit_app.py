import streamlit as st
import numpy as np
from PIL import Image
from collections import Counter
import pandas as pd
import cv2

# Convert RGB to HSV and classify colors into bins

def classify_color_hue(hsv):
    h, s, v = hsv
    if s < 25 or v < 25:
        return "black"
    elif v > 230 and s < 30:
        return "white"
    elif s < 30:
        return "gray"
    if 0 <= h <= 15 or h >= 165:
        return "red"
    elif 16 <= h <= 45:
        return "orange"
    elif 46 <= h <= 75:
        return "yellow"
    elif 76 <= h <= 150:
        return "green"
    elif 151 <= h <= 200:
        return "cyan"
    elif 201 <= h <= 250:
        return "blue"
    elif 251 <= h <= 300:
        return "purple"
    else:
        return "pink"

st.title("🔬 High-Sensitivity Color Classification for Medical Imaging")

uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png", "tif", "tiff"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing pixel color categories with medical-level precision..."):
        img_np = np.array(image)
        h, w, _ = img_np.shape

        # Convert RGB to HSV
        hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        hsv_reshaped = hsv_img.reshape(-1, 3)

        # Classify by perceptual hue ranges
        labels = [classify_color_hue(pixel) for pixel in hsv_reshaped]
        counter = Counter(labels)
        total = sum(counter.values())

        # Prepare particle table
        data_rows = []
        for name, count in counter.items():
            mask = np.array(labels) == name
            selected_pixels = img_np.reshape(-1, 3)[mask]
            mean_intensity = int(np.mean(selected_pixels)) if len(selected_pixels) > 0 else 0
            data_rows.append({
                "Particle": f"CTL LLLOS 467.tif ({name})",
                "Count": count,
                "Total Area": float(count),
                "Average Size": 1.0,
                "%Area": round((count / total) * 100, 3),
                "Mean": mean_intensity
            })

        df = pd.DataFrame(data_rows).sort_values("%Area", ascending=False)

    st.subheader("📊 Particle Summary Table (Hue-Classified)")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "hue_classified_summary.csv", "text/csv")

