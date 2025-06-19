import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from skimage.measure import label, regionprops
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
from skimage.color import label2rgb

def closest_color_name(rgb_tuple):
    css3_colors = {
        'red': (255, 0, 0), 'green': (0, 128, 0), 'blue': (0, 0, 255),
        'black': (0, 0, 0), 'white': (255, 255, 255), 'gray': (128, 128, 128),
        'yellow': (255, 255, 0), 'cyan': (0, 255, 255), 'magenta': (255, 0, 255),
        'orange': (255, 165, 0), 'pink': (255, 192, 203), 'purple': (128, 0, 128),
        'brown': (165, 42, 42), 'lime': (0, 255, 0), 'navy': (0, 0, 128),
        'maroon': (128, 0, 0), 'olive': (128, 128, 0), 'teal': (0, 128, 128),
        'silver': (192, 192, 192)
    }
    min_dist = float('inf')
    closest_name = None
    for name, rgb in css3_colors.items():
        dist = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb_tuple, rgb)) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def analyze_particles(image: Image.Image, filename: str, min_area: int = 10, n_color_groups: int = 6):
    img_rgb = np.array(image.convert("RGB"))
    img_gray = np.array(image.convert("L"))

    threshold = threshold_otsu(img_gray)
    binary = img_gray > threshold

    labeled = label(binary)
    props = regionprops(labeled, intensity_image=img_gray)

    particles = []
    for prop in props:
        if prop.area < min_area:
            continue
        coords = prop.coords
        avg_color = img_rgb[coords[:, 0], coords[:, 1]].mean(axis=0)
        particles.append({
            "coords": coords,
            "area": prop.area,
            "avg_color": avg_color
        })

    if not particles:
        return [], 0, None

    avg_colors = np.array([p["avg_color"] for p in particles])
    lab_colors = rgb2lab(avg_colors[np.newaxis, :, :]/255.0)[0]

    kmeans = KMeans(n_clusters=min(n_color_groups, len(particles)), random_state=42)
    labels = kmeans.fit_predict(lab_colors)

    groups = defaultdict(lambda: {"count": 0, "total_area": 0, "mean_color_sum": np.array([0, 0, 0], dtype=float)})

    for i, label_id in enumerate(labels):
        p = particles[i]
        groups[label_id]["count"] += 1
        groups[label_id]["total_area"] += p["area"]
        groups[label_id]["mean_color_sum"] += p["avg_color"] * p["area"]

    total_image_area = img_rgb.shape[0] * img_rgb.shape[1]
    rows = []

    for group_id, data in groups.items():
        count = data["count"]
        total_area = data["total_area"]
        avg_size = total_area / count
        percent_area = (total_area / total_image_area) * 100
        mean_color = (data["mean_color_sum"] / total_area).astype(int)
        color_name = closest_color_name(mean_color)
        mean_intensity = int(np.mean(mean_color))

        rows.append({
            "Label": f"{filename} ({color_name})",
            "Color": color_name,
            "Count": count,
            "Total Area": total_area,
            "Average Size": round(avg_size, 3),
            "Percentage Area": round(percent_area, 3),
            "Mean Intensity": mean_intensity
        })

    overlay = label2rgb(labeled, image=img_rgb, bg_label=0)

    return rows, len(particles), overlay

# --- Streamlit App ---
st.title("🔬 Particle Analyzer (Accurate Color & Shape Detection)")
st.write("Upload a high-resolution image. We'll detect each particle, group by color, and measure the stats for each group.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    min_area = st.slider("Minimum Particle Area (filter small dots)", 1, 1000, 30)
    with st.spinner("Detecting particles..."):
        rows, count, overlay = analyze_particles(image, uploaded_file.name, min_area=min_area)

    st.subheader("🧾 Total Particles Detected")
    st.write(f"**{count}** particles found.")

    if rows:
        st.subheader("📊 Particle Group Summary")
        st.dataframe(pd.DataFrame(rows))
    else:
        st.warning("No valid particles found.")

    if overlay is not None:
        st.subheader("🖼️ Overlay Preview")
        st.image((overlay * 255).astype(np.uint8), use_column_width=True)
