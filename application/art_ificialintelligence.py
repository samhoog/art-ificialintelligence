from fastai.vision.all import *
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Art-ificial Intelligence")
st.write(os.getcwd())
os.chdir('application')
st.write(os.getcwd())
st.write(listdir())
classifier = load_learner('doodle_classifier.pkl')

stroke_width = st.sidebar.slider("Stroke width: ", 10, 25, 15)

realtime_update = st.sidebar.checkbox("Update in realtime", True)

categories = ['bicycle', 'cactus', 'camera', 'flower', 'helicopter', 'monkey', 'moon',
              'mug', 'octopus', 'owl', 'panda', 'shorts', 'strawberry', 'submarine',
              'sword', 'telephone', 'tree', 'truck', 'violin', 'windmill']

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color='white',
    background_color='black',
    update_streamlit=realtime_update,
    height=448,
    width=448,
    drawing_mode="freedraw",
    key="canvas",
)

# Convert the image to a 28 x 28 numpy
img = Image.fromarray(canvas_result.image_data)
img = img.resize(size=(28, 28)).convert('RGB')
img = np.array(img)

# Predict what the image is
prediction = classifier.predict(img)

# Calculate the guess and highest percentage and put them on the screen
percentages = prediction[2].numpy()
max_percentage = percentages.max()
max_percentage = round(max_percentage * 100, 3)
st.write('I think it is a ' + str(prediction[0]))
st.write('I am ' + str(max_percentage) + '% confident!')

# Calculate the percentages and put them on the sidebar
percentages = (percentages * 100).tolist()
percentages = [round(percentage, 3) for percentage in percentages]
res = {categories[i]: percentages[i] for i in range(len(categories))}
with st.sidebar:
    st.write(res)
