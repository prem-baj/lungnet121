from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io

app = Flask(__name__)

if torch.cuda.is_available():
    device=torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Load the model weights
model_path = "covid_densenet121.pth"
class_names = ['covid19', 'normal']

def CNN_Model(weights=True):
    model = models.densenet121(weights=weights) # Returns Defined Densenet model with weights trained on ImageNet
    # models.resnet50(pretrained=True)
    num_ftrs = model.classifier.in_features # Get the number of features output from CNN layer
    model.classifier = nn.Linear(num_ftrs, len(class_names)) # Overwrites the Classifier layer with custom defined layer for transfer learning
    model = model.to(device) # Transfer the Model to GPU if available

    return model

model = CNN_Model(weights=True)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the class labels

# Define the image transformations
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]
test_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_nums, std=std_nums)
])

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image
    img = request.files['image'].read()
    img = Image.open(io.BytesIO(img))
    img = img.convert('RGB')
    # Transform the image
    img_tensor = test_transforms(img)
    img_tensor = img_tensor.unsqueeze(0)  # add batch dimension

    # Predict the class of the image
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.log_softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        class_label = class_names[class_idx]

    # Return the result
    return render_template('./index.html', prediction=class_label)

if __name__ == '__main__':
    app.run(debug=True)
