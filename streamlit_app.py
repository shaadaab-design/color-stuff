import streamlit as st
import numpy as np
from collections import defaultdict
from PIL import Image
import pandas as pd

from skimage import measure, color
from skimage.filters import threshold_otsu
from skimage.color import label2rgb

def dominant_color_name(rgb_tuple):
    r, g, b = rgb_tuple
    if r > g and r > b:
        return "red"
    elif g > r and g > b:
        return "green"
    elif b > r and b > g:
        return "blue"
    elif abs(r - g) < 10 and abs(g - b) < 10:
        return "gray"
    elif g > 100 and b > 100:
        return "teal"
    elif r > 100 and g > 100:
        return "yellow" if b < 100 else "white"
    elif r > 100 and b > 100:
        return "magenta"
    else:
        return "unknown"

def particle_analysis_dominant_color(image, filename, min_area=10, threshold=None):
    gray_img = image.convert("L")
    img_np_gray = np.array(gray_img)
    img_np_color = np.array(image.convert("RGB"))

    if threshold is None:
        threshold = threshold_otsu(img_np_gray)
    binary = img_np_gray > threshold

    labels_img = measure.label(binary)
    props = measure.regionprops(labels_img, intensity_image=img_np_gray)

    total_image_area = img_np_gray.shape[0] * img_np_gray.shape[1]
    data_groups = defaultdict(lambda: {"count": 0, "total_area": 0, "mean_color_sum": np.array([0, 0, 0], dtype=float)})

    for prop in props:
        if prop.area < min_area:
            continue
        coords = prop.coords
        mean_color = img_np_color[coords[:, 0], coords[:, 1]].mean(axis=0)
        color_name = dominant_color_name(mean_color)
        data_groups[color_name]["count"] += 1
        data_groups[color_name]["total_area"] += prop.area
        data_groups[color_name]["mean_color_sum"] += mean_color * prop.area

    data_rows = []
    for color_name, data in data_groups.items():
        count = data["count"]
        total_area = data["total_area"]
        avg_size = total_area / count if count else 0
        percent_area = (total_area / total_image_area) * 100
        mean_color = (data["mean_color_sum"] / total_area).astype(int)
        mean_intensity = int(np.mean(mean_color))

        label = f"{filename} ({color_name})"
        data_rows.append({
            "Label": label,
            "Color": color_name,
            "Count": count,
            "Total Area": round(total_area, 3),
            "Average Size": round(avg_size, 3),
            "Percentage Area": round(percent_area, 3),
            "Mean Intensity": mean_intensity
        })

    overlay = label2rgb(labels_img, image=np.array(image), bg_label=0)
    return data_rows, len(props), overlay

# --- STREAMLIT UI ---
st.title("🎯 Accurate Particle Analyzer (Color by Channel Dominance)")
st.write("Upload a particle image to detect and summarize by RED, GREEN, BLUE dominance.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.subheader("⚙️ Detection Settings")
    min_area = st.slider("Minimum Particle Area", 1, 1000, 20)
    manual_thresh = st.slider("Manual Threshold (0 = auto)", 0, 255, 0)
    actual_thresh = None if manual_thresh == 0 else manual_thresh

    with st.spinner("Analyzing particles..."):
        rows, total_particles, overlay = particle_analysis_dominant_color(
            image,
            uploaded_file.name,
            min_area=min_area,
            threshold=actual_thresh
        )

    st.subheader("🧾 Total Particles Detected")
    st.write(f"**{total_particles}** particles (before grouping).")

    if overlay is not None:
        st.subheader("🖼️ Particle Overlay Preview")
        st.image((overlay * 255).astype(np.uint8), caption="Overlay", use_container_width=True)

    if rows:
        df = pd.DataFrame(rows)
        st.subheader("📊 Summary by Color Group")
        st.dataframe(df)
    else:
        st.warning("No particles found. Try adjusting threshold or area filter.")

