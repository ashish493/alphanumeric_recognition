import numpy as np
from PIL import Image
import torch
from .numpy_network import NumpyNet

class NumpyModel:
    """
    NumPy-based model wrapper that provides the same interface as MyModel
    but uses pure NumPy implementation for inference
    """
    
    def __init__(self, model_weights: str, device: str):
        """
        Initialize the NumPy model
        
        Args:
            model_weights: Path to the PyTorch weights file
            device: Device specification (ignored for NumPy implementation)
        """
        self.net = NumpyNet()
        self.weights_path = model_weights
        self._initialize()
    
    def _initialize(self):
        """Load weights from PyTorch model and convert to NumPy"""
        try:
            # Load PyTorch weights
            checkpoint = torch.load(self.weights_path, map_location='cpu')
            
            # Extract state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Convert and load weights into NumPy network
            self.net.load_weights_from_pytorch(state_dict)
            
            print("Successfully loaded PyTorch weights into NumPy model")
            
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Using randomly initialized weights")
    
    def predict(self, path):
        """
        Predict using the NumPy implementation
        
        Args:
            path: Path to image file or BytesIO object
            
        Returns:
            numpy array of probabilities for each class
        """
        # Open and preprocess the image
        img = Image.open(path).convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Invert colors (white background to black, black text to white)
        img_array = 1.0 - img_array
        
        # Reshape to (1, 1, 28, 28) for batch processing
        img_tensor = img_array.reshape(1, 1, 28, 28)
        
        # Run prediction
        probabilities = self.net.predict(img_tensor)
        
        # Return probabilities for the single image
        return probabilities[0]
    
    def convert_pytorch_to_numpy_weights(self, pytorch_weights_path: str, numpy_weights_path: str):
        """
        Convert PyTorch weights to NumPy format and save
        
        Args:
            pytorch_weights_path: Path to PyTorch weights file
            numpy_weights_path: Path where to save NumPy weights
        """
        try:
            # Load PyTorch weights
            checkpoint = torch.load(pytorch_weights_path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load into NumPy network
            self.net.load_weights_from_pytorch(state_dict)
            
            # Save as NumPy weights
            self.net.save_weights(numpy_weights_path)
            
            print(f"Successfully converted and saved weights to {numpy_weights_path}")
            
        except Exception as e:
            print(f"Error converting weights: {e}")
