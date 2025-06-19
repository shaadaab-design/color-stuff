import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

def closest_color_name(rgb_tuple):
    css3_names_to_rgb = {
        'red': (255, 0, 0), 'green': (0, 128, 0), 'blue': (0, 0, 255), 'black': (0, 0, 0),
        'white': (255, 255, 255), 'gray': (128, 128, 128), 'yellow': (255, 255, 0),
        'cyan': (0, 255, 255), 'magenta': (255, 0, 255), 'orange': (255, 165, 0),
        'pink': (255, 192, 203), 'purple': (128, 0, 128), 'brown': (165, 42, 42),
        'lime': (0, 255, 0), 'navy': (0, 0, 128), 'maroon': (128, 0, 0),
        'olive': (128, 128, 0), 'teal': (0, 128, 128), 'silver': (192, 192, 192),
    }
    min_dist = float("inf")
    closest = None
    for name, rgb in css3_names_to_rgb.items():
        dist = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb_tuple, rgb)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest = name
    return closest

def analyze_image_by_color_clusters(image, filename, n_clusters=6):
    image = image.convert("RGB")
    img_small = image.resize((image.width // 2, image.height // 2))  # Resize for speed
    img_array = np.array(img_small)
    pixels = img_array.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(int)

    counts = Counter(labels)
    total_pixels = sum(counts.values())

    rows = []
    label_img = np.array([centers[label] for label in labels], dtype=np.uint8).reshape(img_array.shape)

    for i, center in enumerate(centers):
        color_name = closest_color_name(center)
        count = counts[i]
        percent_area = (count / total_pixels) * 100
        mean_intensity = int(np.mean(center))
        rows.append({
            "Label": f"{filename} ({color_name})",
            "Color": color_name,
            "Count": count,
            "Total Area": count,
            "Average Size": 1,  # Pixel-level
            "Percentage Area": round(percent_area, 2),
            "Mean Intensity": mean_intensity
        })

    return rows, label_img

# --- Streamlit App ---
st.title("🎨 Accurate Color Cluster Analyzer (No Thresholds Needed)")
st.write("Upload any image to analyze the most dominant color regions without needing to adjust settings.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Uploaded Image", use_column_width=True)

    with st.spinner("Clustering colors..."):
        rows, clustered_img = analyze_image_by_color_clusters(image, uploaded_file.name, n_clusters=6)

    st.subheader("📊 Color Cluster Summary")
    df = pd.DataFrame(rows)
    st.dataframe(df)

    st.subheader("🖼️ Clustered Image Preview")
    st.image(clustered_img, caption="Clustered Color Regions", use_column_width=True)

