import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

from skimage import measure
from skimage.filters import threshold_otsu

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)

def get_color_percentages(image, num_colors=5):
    img = image.resize((200, 200))  # Resize to speed up processing
    img_np = np.array(img)
    img_np = img_np.reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors)
    labels = kmeans.fit_predict(img_np)
    counts = Counter(labels)
    total = sum(counts.values())

    colors = kmeans.cluster_centers_.astype(int)

    results = []
    for idx, count in counts.items():
        percent = count / total * 100
        hex_color = rgb_to_hex(colors[idx])
        results.append((hex_color, percent))

    return results, colors, counts

def particle_analysis(image):
    # Convert to grayscale
    gray_img = image.convert("L")
    img_np = np.array(gray_img)

    # Threshold to binary image
    thresh = threshold_otsu(img_np)
    binary = img_np > thresh

    # Label connected components (particles)
    labels = measure.label(binary)
    props = measure.regionprops(labels, intensity_image=img_np)

    count = len(props)
    total_area = sum([prop.area for prop in props])
    avg_size = total_area / count if count > 0 else 0
    percent_area = total_area / (img_np.shape[0] * img_np.shape[1]) * 100

    # Mean intensity per particle
    means = [prop.mean_intensity for prop in props]

    return {
        "count": count,
        "total_area": total_area,
        "average_size": avg_size,
        "percent_area": percent_area,
        "mean_intensity": means,
        "props": props
    }

st.title("🎨 Color & Particle Analyzer")
st.write("Upload any image (including high-quality TIFF) to analyze color percentages and particle statistics.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing colors..."):
        results, colors, counts = get_color_percentages(image, num_colors=6)

    st.subheader("📊 Color Breakdown")
    for hex_color, percent in results:
        st.markdown(f"<div style='background-color:{hex_color}; padding:10px; color:white;'>{hex_color}: {percent:.2f}%</div>", unsafe_allow_html=True)

    st.subheader("Pie Chart of Dominant Colors")
    fig, ax = plt.subplots()
    plt.pie([p for _, p in results], 
            labels=[c for c, _ in results],
            colors=[c for c, _ in results],
            autopct='%1.1f%%')
    st.pyplot(fig)

    with st.spinner("Analyzing particles..."):
        stats = particle_analysis(image)

    st.subheader("🧩 Particle Analysis")
    st.write(f"Count: {stats['count']}")
    st.write(f"Total Area (pixels): {stats['total_area']}")
    st.write(f"Average Size (pixels): {stats['average_size']:.2f}")
    st.write(f"Percentage Area: {stats['percent_area']:.2f}%")

    st.write("Mean intensity of each particle:")
    for i, mean_val in enumerate(stats["mean_intensity"], 1):
        st.write(f"Particle {i}: {mean_val:.2f}")
