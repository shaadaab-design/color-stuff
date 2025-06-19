import streamlit as st
import numpy as np
from PIL import Image
from collections import Counter
import pandas as pd
from skimage.measure import label, regionprops
from skimage.color import rgb2lab, deltaE_cie76
import matplotlib.pyplot as plt

# Define reference colors in RGB and convert to Lab
hue_ref_colors = {
    "red": (255, 0, 0),
    "orange": (255, 165, 0),
    "yellow": (255, 255, 0),
    "green": (0, 255, 0),
    "cyan": (0, 255, 255),
    "blue": (0, 0, 255),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "white": (255, 255, 255)
}

ref_lab = {k: rgb2lab(np.uint8([[v]]))[0][0] for k, v in hue_ref_colors.items()}

st.title("🔬 DeltaE-Based Perceptual Color Classification (CIE-Lab)")

uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png", "tif", "tiff"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    threshold = st.slider("DeltaE Sensitivity Threshold", 5, 60, 25)

    with st.spinner("Classifying pixels using perceptual Lab color model and DeltaE threshold..."):
        img_np = np.array(image)
        h, w, _ = img_np.shape
        flat_pixels = img_np.reshape(-1, 3)
        lab_pixels = rgb2lab(flat_pixels.reshape(-1, 1, 3)).reshape(-1, 3)

        # Multi-match classification based on threshold
        pixel_labels = []
        for lab_pixel in lab_pixels:
            best_label = None
            best_distance = float("inf")
            for name, ref in ref_lab.items():
                dist = np.linalg.norm(lab_pixel - ref)
                if dist < threshold and dist < best_distance:
                    best_label = name
                    best_distance = dist
            pixel_labels.append(best_label if best_label else "unclassified")

        counter = Counter(pixel_labels)
        total = sum(counter.values())

        # Particle summary table
        data_rows = []
        for name, count in counter.items():
            if name == "unclassified":
                continue
            mask = np.array(pixel_labels) == name
            selected_pixels = flat_pixels[mask]
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

    st.subheader("📊 Particle Summary Table (DeltaE-Classified)")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "deltaE_classified_summary.csv", "text/csv")

    # Reconstructed image using label colors
    st.subheader("🖼️ Perceptually Reconstructed Image")
    reconstructed_img = np.zeros((h * w, 3), dtype=np.uint8)
    for i, name in enumerate(pixel_labels):
        reconstructed_img[i] = hue_ref_colors.get(name, (0, 0, 0))
    reconstructed_img = reconstructed_img.reshape((h, w, 3))
    st.image(Image.fromarray(reconstructed_img), caption="DeltaE Reconstructed Image", use_column_width=True)

    # Region overlay for selected hue
    st.subheader("🧪 Region Detection by DeltaE")
    available_labels = [l for l in df["Particle"].str.extract(r'\((.*?)\)')[0].unique() if l != "unclassified"]
    selected_hue = st.selectbox("Select color to view region map", available_labels)

    label_array = np.array(pixel_labels).reshape(h, w)
    binary_mask = (label_array == selected_hue).astype(np.uint8)
    labeled = label(binary_mask)
    props = regionprops(labeled)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    for region in props:
        y0, x0, y1, x1 = region.bbox
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='red', facecolor='none', linewidth=1)
        ax.add_patch(rect)
    ax.set_title(f"Detected regions for hue: {selected_hue}")
    ax.axis("off")
    st.pyplot(fig)

