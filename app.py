import torch
import torchvision
from torchvision import transforms, datasets
import streamlit as st
from utils.model import efficientnetb0
from PIL import Image

# membuat judul
st.title("Klasifikasi Permasalahan Kulit")

# membuat nama kelas
class_names = ['kulit belang', 'kulit normal', 'sunburn']

# load model
model = efficientnetb0(OUT_FEATURES=len(class_names))
model.load_state_dict(torch.load("models/model_tuning.pth", map_location=torch.device('cpu')))

# membuat evaluasi model
model.eval()

# membuat data transforms untuk klasifikasi
data_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)), # Mengubah ukuran gambar menjadi Lebar = 224 pixel, Tinggi=224 pixel
    transforms.ToTensor(), # mengubah data dari jpeg, jpg, png ke Tensor()
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]) # Mengubah menjadi standar ImageNet
])

# membuat fungsi untuk input file
uploader_file = st.file_uploader('Upload an image', type=['jpg','jpeg','png'])


# Membuat kondisi untuk input file
if uploader_file is not None:

    # membaca gambar
    image = Image.open(uploader_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # mengubah gambar
    image_tensor = data_transforms(image).unsqueeze(0)

    # membuat prediksi
    model.eval()
    with torch.inference_mode():
        output = model(image_tensor)
        prob = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(prob, dim=1).item()

    # menampilkan hasil prediksi
    st.write(f'Predicted Class: {class_names[predicted_class]} \nProbability: {prob[0][predicted_class]:.4f}')


# membuat kondisi untuk input menggunakan kamera
if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

# Create condition for camera input
if st.button('Take a picture'):
    st.session_state.show_camera = True

# Create condition if the camera input is true
if st.session_state.show_camera:

    # Capture the image from camera
    foto = st.camera_input('Take a Picture')

    # Condition if the image is captured
    if foto is not None:

        # Open the captured image
        foto = Image.open(foto)

        # Display the capture image
        st.image(foto, caption='Captured Image', use_container_width=True)

        # Preprocess the captured image
        foto_tensor = data_transforms(foto).unsqueeze(0)

        # Make predictions
        model.eval()
        with torch.inference_mode():
            output = model(foto_tensor)
            prob = torch.softmax(output, dim=1)
            prediction_class = torch.argmax(prob, dim=1).item()

        # Display the prediction class
        st.write(f'Predicted Class: {class_names[prediction_class]} \nProbability: {prob[0][prediction_class]:.4f}')

    # Create a button to close the camera input
    if st.button('Close Camera'):
        st.session_state.show_camera = False