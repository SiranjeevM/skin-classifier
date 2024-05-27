import streamlit as st
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np
st.title("Skin Disease Classification")

st.write("Upload the images of Hair Loss Photos Alopecia and other Hair Diseases,Melanoma Skin Cancer Nevi and Moles,Nail Fungus and other Nail Disease,Urticaria Hives,Warts Molluscum and other Viral Infections.")

model = load_model("model.h5",custom_objects={'KerasLayer':hub.KerasLayer})
labels = {
      0: 'Hair Loss Photos Alopecia and other Hair Diseases',
    1: 'Melanoma Skin Cancer Nevi and Moles',
    2: 'Nail Fungus and other Nail Disease',
    3: 'Urticaria Hives',
    4: 'Warts Molluscum and other Viral Infections',
}
uploaded_file = st.file_uploader(
    "Upload an image of a Sea Animal:", type=['jpg','png','jpeg']
)
predictions=-1
if uploaded_file is not None:
    image1 = Image.open(uploaded_file)
    image1=image.smart_resize(image1,(224,224))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    label=labels[np.argmax(predictions)]


st.write("### Prediction Result")
if st.button("Predict"):
    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        st.image(image1, caption="Uploaded Image", use_column_width=True)
        st.markdown(
            f"<h2 style='text-align: center;'>Image of {label}</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.write("Please upload file or choose sample image.")



# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from PIL import Image

# # Load the trained model
# model = tf.keras.models.load_model('model.h5', custom_objects={'KerasLayer': tf.keras.layers.Layer})

# # Define the class labels
# labels = ['Hair Loss Photos Alopecia and other Hair Diseases', 
#           'Melanoma Skin Cancer Nevi and Moles', 
#           'Nail Fungus and other Nail Disease', 
#           'Urticaria Hives', 
#           'Warts Molluscum and other Viral Infections']

# # Define the title of the app
# st.title('Skin Disease Classifier')

# # Add a file uploader to upload an image
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("")
#     st.write("Classifying...")

#     # Preprocess the image
#     img_shape = (224, 224)
#     image = image.resize(img_shape)
#     image_array = np.array(image) / 255.0
#     image_array = np.expand_dims(image_array, axis=0)

#     # Predict the class of the image
#     result = model.predict(image_array)
#     predicted_class_index = np.argmax(result)
#     predicted_class_probability = result[0][predicted_class_index]

#     # Display the prediction result
#     st.write(f"Predicted Class: {labels[predicted_class_index]}")
#     st.write(f"Confidence: {predicted_class_probability:.2f}")

# # Run the app with: streamlit run app.py
