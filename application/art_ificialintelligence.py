
import numpy as np
import cairocffi as cairo
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import doodle_classification

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)

realtime_update = st.sidebar.checkbox("Update in realtime", True)



# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color='white',
    background_color='black',
    update_streamlit=realtime_update,
    height=500,
    width=500,
    drawing_mode=drawing_mode,
    key="canvas",
)

# Do something interesting with the image data and paths
img = Image.fromarray(canvas_result.image_data)
img = img.resize(size=(28, 28))
st.image(img)

