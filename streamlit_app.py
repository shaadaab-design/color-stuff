import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import io

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

# Streamlit UI
st.title("🎨 Color Percentage Analyzer")
st.write("Upload any image to see what % of colors it contains.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing colors..."):
        results, colors, counts = get_color_percentages(image, num_colors=6)

    st.subheader("📊 Color Breakdown")
    for hex_color, percent in results:
        st.markdown(f"<div style='background-color:{hex_color}; padding:10px; color:white;'>{hex_color}: {percent:.2f}%</div>", unsafe_allow_html=True)

    # Show pie chart
    st.subheader("Pie Chart of Dominant Colors")
    fig, ax = plt.subplots()
    plt.pie([p for _, p in results], 
            labels=[c for c, _ in results],
            colors=[c for c, _ in results],
            autopct='%1.1f%%')
    st.pyplot(fig)