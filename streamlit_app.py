import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from collections import defaultdict

st.set_page_config(page_title="Pixel-Based Color Analyzer", layout="wide")
st.title("🔬 Pixel-Level Accurate Color Analyzer")
st.write("Upload an image to analyze the percentage of different colors based on pixel clustering.")


def closest_css3_color_name(rgb_tuple):
    css3_colors = {
        'red': (255, 0, 0), 'green': (0, 128, 0), 'blue': (0, 0, 255),
        'black': (0, 0, 0), 'white': (255, 255, 255), 'gray': (128, 128, 128),
        'yellow': (255, 255, 0), 'cyan': (0, 255, 255), 'magenta': (255, 0, 255),
        'orange': (255, 165, 0), 'pink': (255, 192, 203), 'purple': (128, 0, 128),
        'brown': (165, 42, 42), 'lime': (0, 255, 0), 'navy': (0, 0, 128),
        'maroon': (128, 0, 0), 'olive': (128, 128, 0), 'teal': (0, 128, 128),
        'silver': (192, 192, 192),
    }
    min_dist = float('inf')
    name = "unknown"
    for cname, rgb in css3_colors.items():
        dist = np.linalg.norm(np.array(rgb) - np.array(rgb_tuple))
        if dist < min_dist:
            min_dist = dist
            name = cname
    return name


def pixel_color_analysis(image, filename, n_clusters=6):
    img_rgb = image.convert("RGB")
    img_array = np.array(img_rgb)
    flat_pixels = img_array.reshape(-1, 3)

    # Cluster pixels
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(flat_pixels)
    centers = kmeans.cluster_centers_.astype(int)

    total_pixels = flat_pixels.shape[0]
    results = []
    for i in range(n_clusters):
        mask = labels == i
        count = np.sum(mask)
        percent = round((count / total_pixels) * 100, 2)
        color = centers[i]
        mean_intensity = int(np.mean(color))
        name = closest_css3_color_name(color)
        results.append({
            "Label": f"{filename} ({name})",
            "Color": name,
            "Count (pixels)": int(count),
            "Percentage Area": percent,
            "Mean RGB": str(tuple(color)),
            "Mean Intensity": mean_intensity
        })

    return pd.DataFrame(results)


uploaded_file = st.file_uploader("📤 Upload your image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    clusters = st.slider("Number of Color Clusters", 2, 10, 6)

    with st.spinner("Analyzing pixels..."):
        df = pixel_color_analysis(image, uploaded_file.name, n_clusters=clusters)

    st.subheader("📊 Color Cluster Results")
    st.dataframe(df)

    st.download_button("Download CSV", df.to_csv(index=False), file_name="pixel_color_results.csv", mime="text/csv")

