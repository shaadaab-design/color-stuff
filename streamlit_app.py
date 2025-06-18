import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from PIL import Image
import pandas as pd
from skimage import measure, color
from skimage.color import rgb2lab, label2rgb

def closest_color_name(rgb_tuple):
    # Common CSS3 colors for matching
    css3_names_to_rgb = {
        'red': (255, 0, 0),
        'green': (0, 128, 0),
        'blue': (0, 0, 255),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'gray': (128, 128, 128),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'orange': (255, 165, 0),
        'pink': (255, 192, 203),
        'purple': (128, 0, 128),
        'brown': (165, 42, 42),
        'lime': (0, 255, 0),
        'navy': (0, 0, 128),
        'maroon': (128, 0, 0),
        'olive': (128, 128, 0),
        'teal': (0, 128, 128),
        'silver': (192, 192, 192),
    }

    min_dist = float("inf")
    closest_name = None
    for name, rgb in css3_names_to_rgb.items():
        dist = np.linalg.norm(np.array(rgb_tuple) - np.array(rgb))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def particle_analysis(image, filename, n_clusters=6, min_area=10):
    # Convert image to numpy arrays
    img_rgb = np.array(image.convert("RGB"))

    # Flatten pixels for clustering
    pixels = img_rgb.reshape(-1, 3)

    # KMeans clustering for dominant colors
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    # Create label image for segmentation
    label_img = labels.reshape(img_rgb.shape[0], img_rgb.shape[1])

    # Analyze each cluster as a group of particles
    data_rows = []
    total_image_area = img_rgb.shape[0] * img_rgb.shape[1]

    for cluster_id, center_color in enumerate(centers):
        mask = label_img == cluster_id
        area = np.sum(mask)
        if area < min_area:
            continue

        percent_area = (area / total_image_area) * 100
        mean_color = center_color.astype(int)
        color_name = closest_color_name(mean_color)
        mean_intensity = int(np.mean(mean_color))

        label = f"{filename} ({color_name})"

        data_rows.append({
            "Label": label,
            "Color": color_name,
            "Count": "-",  # No individual particle count here
            "Total Area": round(area, 2),
            "Average Size": "-",
            "Percentage Area": round(percent_area, 3),
            "Mean Intensity": mean_intensity
        })

    # Generate overlay image with cluster colors
    overlay = label2rgb(label_img, image=img_rgb, bg_label=-1)

    return data_rows, overlay

# --- STREAMLIT UI ---
st.title("🧪 Accurate Particle Color Grouping")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    n_clusters = st.slider("Number of Color Groups (Clusters)", 2, 15, 6)
    min_area = st.slider("Minimum Area (pixels)", 1, 500, 10)

    with st.spinner("Analyzing image colors..."):
        rows, overlay = particle_analysis(image, uploaded_file.name, n_clusters=n_clusters, min_area=min_area)

    if rows:
        df = pd.DataFrame(rows)
        st.subheader("📊 Particle Color Group Summary")
        st.dataframe(df)

        st.subheader("🖼️ Color Group Overlay")
        st.image((overlay * 255).astype(np.uint8), use_container_width=True)
    else:
        st.warning("No significant color groups detected. Try lowering the minimum area or increasing clusters.")

