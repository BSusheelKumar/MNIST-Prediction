from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import base64
from io import BytesIO
from torch import nn
app = Flask(__name__)

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
model = ResNet9(1,10)
# Load the PyTorch model
model.load_state_dict(torch.load('MNIST-ResNet9 (1).pth'))
model.eval()

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Decode the base64 image and convert to PIL Image
    image_data = data['imageData'].split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    
    # Apply the transformation
    transformed_image = transform(image)

    # Add batch dimension and convert to float tensor
    transformed_image = transformed_image.unsqueeze(0).float()

    # Make prediction
    with torch.no_grad():
        output = model(transformed_image)
        _, predicted_class = torch.max(output, 1)

    prediction = int(predicted_class.item())

    # Convert the transformed image to a PIL Image for display
    transformed_image_pil = transforms.ToPILImage()(transformed_image.squeeze())

    # Save the PIL Image to a BytesIO object
    image_bytes = BytesIO()
    transformed_image_pil.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    # Return the response with prediction and transformed image
    return jsonify({
        'prediction': prediction,
        'transformedImage': base64.b64encode(image_bytes.read()).decode('utf-8')
    })

@app.route('/get_transformed_image/<transformed_image_base64>')
def get_transformed_image(transformed_image_base64):
    # Decode the base64 image and return it as a response
    transformed_image_bytes = base64.b64decode(transformed_image_base64)
    return send_file(BytesIO(transformed_image_bytes), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)