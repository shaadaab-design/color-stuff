import streamlit as st
import numpy as np
from PIL import Image
from collections import Counter
import pandas as pd
from skimage.measure import label, regionprops
from skimage.color import rgb2lab
import matplotlib.pyplot as plt

# Define reference RGB colors
hue_ref_colors = {
    "red": [(255, 0, 0)],
    "orange": [(255, 165, 0)],
    "yellow": [(255, 255, 0)],
    "green": [(0, 255, 0), (50, 200, 50), (128, 255, 128), (100, 255, 150)],
    "cyan": [(0, 255, 255)],
    "blue": [(0, 0, 255)],
    "purple": [(128, 0, 128)],
    "pink": [(255, 192, 203)],
    "black": [(0, 0, 0)],
    "gray": [(128, 128, 128)],
    "white": [(255, 255, 255)]
}

# Convert reference colors to Lab
ref_lab = {
    name: [rgb2lab(np.uint8([[rgb]]))[0][0] for rgb in rgbs]
    for name, rgbs in hue_ref_colors.items()
}

st.title("🔬 Advanced Medical Image Color Classification")

uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png", "tif", "tiff"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    threshold = st.slider("DeltaE Sensitivity Threshold", 5, 60, 25)

    img_np = np.array(image)
    h, w, _ = img_np.shape
    flat_pixels = img_np.reshape(-1, 3)
    lab_pixels = rgb2lab(flat_pixels.reshape(-1, 1, 3)).reshape(-1, 3)

    # Optional user-labeled color (interactive feedback)
    st.markdown("### 🎯 Pick a pixel to train a new color reference")
    pick_x = st.number_input("X Pixel", min_value=0, max_value=w - 1, value=0)
    pick_y = st.number_input("Y Pixel", min_value=0, max_value=h - 1, value=0)
    pick_rgb = tuple(img_np[int(pick_y), int(pick_x)])
    pick_lab = rgb2lab(np.uint8([[pick_rgb]]))[0][0]
    st.write(f"Picked RGB: {pick_rgb}")
    new_label = st.text_input("Label this pixel as...", value="custom")
    if new_label:
        if new_label not in ref_lab:
            ref_lab[new_label] = []
        ref_lab[new_label].append(pick_lab)

    # Pixel classification with extended reference set
    pixel_labels = []
    for lab_pixel in lab_pixels:
        best_label = None
        best_distance = float("inf")
        for name, refs in ref_lab.items():
            for ref in refs:
                dist = np.linalg.norm(lab_pixel - ref)
                if dist < threshold and dist < best_distance:
                    best_label = name
                    best_distance = dist
        pixel_labels.append(best_label if best_label else "unclassified")

    counter = Counter(pixel_labels)
    total = sum(counter.values())

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

    st.subheader("📊 Particle Summary Table")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "deltaE_classified_summary.csv", "text/csv")

    # Reconstruct image from labeled colors
    flat_ref_rgb = {k: v[0] for k, v in hue_ref_colors.items() if len(v) > 0}
    st.subheader("🖼️ Reconstructed Image")
    reconstructed_img = np.zeros((h * w, 3), dtype=np.uint8)
    for i, name in enumerate(pixel_labels):
        reconstructed_img[i] = flat_ref_rgb.get(name, (0, 0, 0))
    reconstructed_img = reconstructed_img.reshape((h, w, 3))
    st.image(Image.fromarray(reconstructed_img), caption="DeltaE Reconstructed Image", use_column_width=True)

    # AI anomaly detection: least frequent label mask
    st.subheader("🚨 Anomaly Detection")
    rare_labels = [k for k, v in counter.items() if v < 0.01 * total and k != "unclassified"]
    st.write("Rare particle types (potential anomalies):", rare_labels)

    for label_name in rare_labels:
        label_array = np.array(pixel_labels).reshape(h, w)
        binary_mask = (label_array == label_name).astype(np.uint8)
        labeled = label(binary_mask)
        props = regionprops(labeled)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_np)
        for region in props:
            y0, x0, y1, x1 = region.bbox
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='red', facecolor='none', linewidth=1)
            ax.add_patch(rect)
        ax.set_title(f"Anomaly Regions: {label_name}")
        ax.axis("off")
        st.pyplot(fig)
