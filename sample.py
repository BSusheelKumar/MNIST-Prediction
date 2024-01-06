import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64

# Streamlit app
st.set_page_config(page_title="MNIST Image Prediction", page_icon="✏️")

st.title("MNIST Image Prediction")

# Create a drawing area using st.image
canvas = st.image([0], width=280, channels="RGB", caption="Draw here")

# Predict button
if st.button("Predict"):
    # Convert the canvas drawing to image data
    image_data = canvas.image_data.astype("uint8")

    # Convert to PIL Image
    image = Image.fromarray(image_data)

    # Convert PIL Image to bytes
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    # Convert bytes to base64
    image_data_uri = f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"

    # Send the image data to the server for prediction
    # Replace '/predict' with the correct endpoint for handling predictions on your server
    response = requests.post("/predict", json={"imageData": image_data_uri})
    prediction_data = response.json()

    # Display prediction result
    st.write(f"Prediction: {prediction_data['prediction']}")

    # Display drawn image
    st.image(image_bytes, caption="Drawn Image", use_column_width=True, channels="RGB")

    # Display transformed image
    transformed_image = Image.open(BytesIO(base64.b64decode(prediction_data["transformedImage"])))
    st.image(transformed_image, caption="Transformed Image", use_column_width=True)

# Clear button
if st.button("Clear"):
    # Clear the canvas by updating its value
    canvas.image([0])

# Add some styling
st.markdown(
    """
    <style>
        body {
            background-color: black;
            color: white;
        }

        div[data-baseweb="button"] {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
