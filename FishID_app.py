import streamlit as st
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow import keras
import numpy as np

# regulations dictionary
regulations = {
    0: {"Name": "Mahi-Mahi",
        "Image": "mahi-mahi.png",
        "Florida - Atlantic": {"Season": "Open",
                               "Bag Limit/Angler": 5,
                               "Minimum Size": '20"',
                               "Maximum Size": "None",
                               "Measurement": "Fork Length",
                               "Vessel Limit": 30
            },
        "Florida - Gulf": {"Season": "Open",
                            "Bag Limit/Angler": 10,
                            "Minimum Size": 'None',
                            "Maximum Size": "None",
                            "Measurement": "Fork Length",
                            "Vessel Limit": 60
            }
        },
    1: {"Name": "Sailfish",
        "Image": "sailfish.jpeg",
        "Florida - Atlantic": {"Season": "Open",
                               "Bag Limit/Angler": 1,
                               "Minimum Size": '63"',
                               "Measurement": "Lower Jaw to Fork",
                               "Maximum Size": "None",
                               "Vessel Limit": 1
            },
        "Florida - Gulf": {"Season": "Open",
                               "Bag Limit/Angler": 1,
                               "Minimum Size": '63"',
                               "Measurement": "Lower Jaw to Fork",
                               "Maximum Size": "None",
                               "Vessel Limit": 1
            }
        }
}

# load image classification model
model = tf.keras.models.load_model('FishID_model.h5')

# Title
st.image("FishID_Logo.png")
st.write("""
This app is for demo purposes only. The information may not be accurate and is not inteded to be used as a guide on local fishing regulations.
The app is currently only setup to identify sailfish and mahi-mahi.
""")

# Get region input values
def get_region():
    region = st.sidebar.selectbox("Fishing Region", ["Florida - Atlantic", "Florida - Gulf"])
    return region

region = get_region()

st.write(f"You are currently fishing in the {region} region")

fish_image = None

# take_photo = st.button("ID Fish")
# if take_photo:
fish_image = st.camera_input('')
if fish_image:
    #st.image(fish_image)
    img = image.load_img(fish_image, target_size = (160, 160))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    prediction = model.predict(img)
    prediction = tf.nn.sigmoid(prediction)
    prediction = tf.where(prediction < 0.5, 0, 1)

    species = int(prediction[0][0])
    st.write(f"Congratulation! You caught a {regulations[species]['Name']}!")

    st.image(regulations[species]['Image'])

    st.write(f"{regulations[species]['Name']} regulations for the {region} region:")
    for k, v in regulations[species][region].items():
        st.write(f"{k}: {v}")

    st.write('')
    st.write('Press the "Clear photo" button above to restart')