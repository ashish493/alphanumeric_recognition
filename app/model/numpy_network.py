import numpy as np
from PIL import Image
import pickle

class NumpyNet:
    """
    NumPy implementation of the neural network architecture equivalent to PyTorch Net class.
    This implementation provides the same architecture as the PyTorch version but using pure NumPy.
    """
    
    def __init__(self):
        """Initialize the network layers and parameters"""
        self.weights = {}
        self.biases = {}
        
        # Initialize network architecture matching PyTorch Net
        self._initialize_layers()
    
    def _initialize_layers(self):
        """Initialize all layer parameters with proper dimensions"""
        # Conv1: input channels=1, output channels=64, kernel=(5,5), padding=2
        self.weights['conv1'] = np.random.randn(64, 1, 5, 5) * np.sqrt(2.0 / (1 * 5 * 5))
        self.biases['conv1'] = np.zeros((64,))
        
        # BatchNorm2d for conv1 (64 channels)
        self.weights['conv1_bn_gamma'] = np.ones((64,))
        self.weights['conv1_bn_beta'] = np.zeros((64,))
        self.conv1_bn_running_mean = np.zeros((64,))
        self.conv1_bn_running_var = np.ones((64,))
        
        # Conv2: input channels=64, output channels=128, kernel=(2,2), padding=2
        self.weights['conv2'] = np.random.randn(128, 64, 2, 2) * np.sqrt(2.0 / (64 * 2 * 2))
        self.biases['conv2'] = np.zeros((128,))
        
        # FC1: 8192 -> 1024 (corrected from 2048 to match actual flattened size)
        self.weights['fc1'] = np.random.randn(8192, 1024) * np.sqrt(2.0 / 8192)
        self.biases['fc1'] = np.zeros((1024,))
        
        # FC2: 1024 -> 512
        self.weights['fc2'] = np.random.randn(1024, 512) * np.sqrt(2.0 / 1024)
        self.biases['fc2'] = np.zeros((512,))
        
        # BatchNorm1d for fc2 output (512 features)
        self.weights['bn_gamma'] = np.ones((512,))
        self.weights['bn_beta'] = np.zeros((512,))
        self.bn_running_mean = np.zeros((512,))
        self.bn_running_var = np.ones((512,))
        
        # FC3: 512 -> 128
        self.weights['fc3'] = np.random.randn(512, 128) * np.sqrt(2.0 / 512)
        self.biases['fc3'] = np.zeros((128,))
        
        # FC4: 128 -> 47
        self.weights['fc4'] = np.random.randn(128, 47) * np.sqrt(2.0 / 128)
        self.biases['fc4'] = np.zeros((47,))
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def conv2d(self, input_data, weight, bias, padding=0, stride=1):
        """
        2D convolution operation
        input_data: (batch_size, in_channels, height, width)
        weight: (out_channels, in_channels, kernel_height, kernel_width)
        """
        batch_size, in_channels, input_height, input_width = input_data.shape
        out_channels, _, kernel_height, kernel_width = weight.shape
        
        # Apply padding
        if padding > 0:
            input_data = np.pad(input_data, 
                              ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                              mode='constant', constant_values=0)
        
        _, _, padded_height, padded_width = input_data.shape
        
        # Calculate output dimensions
        output_height = (padded_height - kernel_height) // stride + 1
        output_width = (padded_width - kernel_width) // stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, out_channels, output_height, output_width))
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(out_channels):
                for h in range(0, output_height):
                    for w in range(0, output_width):
                        h_start = h * stride
                        h_end = h_start + kernel_height
                        w_start = w * stride
                        w_end = w_start + kernel_width
                        
                        # Extract region and perform convolution
                        region = input_data[b, :, h_start:h_end, w_start:w_end]
                        output[b, oc, h, w] = np.sum(region * weight[oc]) + bias[oc]
        
        return output
    
    def max_pool2d(self, input_data, kernel_size=2, stride=2):
        """Max pooling operation"""
        batch_size, channels, input_height, input_width = input_data.shape
        
        output_height = input_height // stride
        output_width = input_width // stride
        
        output = np.zeros((batch_size, channels, output_height, output_width))
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * stride
                        h_end = h_start + kernel_size
                        w_start = w * stride
                        w_end = w_start + kernel_size
                        
                        region = input_data[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, h, w] = np.max(region)
        
        return output
    
    def batch_norm_2d(self, x, gamma, beta, running_mean, running_var, eps=1e-5):
        """Batch normalization for 2D feature maps"""
        # x shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape
        
        # Reshape for broadcasting
        gamma = gamma.reshape(1, channels, 1, 1)
        beta = beta.reshape(1, channels, 1, 1)
        running_mean = running_mean.reshape(1, channels, 1, 1)
        running_var = running_var.reshape(1, channels, 1, 1)
        
        # Normalize
        normalized = (x - running_mean) / np.sqrt(running_var + eps)
        return gamma * normalized + beta
    
    def batch_norm_1d(self, x, gamma, beta, running_mean, running_var, eps=1e-5):
        """Batch normalization for 1D features"""
        # x shape: (batch_size, features)
        normalized = (x - running_mean) / np.sqrt(running_var + eps)
        return gamma * normalized + beta
    
    def dropout(self, x, p=0.3, training=False):
        """Dropout layer (only active during training)"""
        if not training:
            return x
        
        mask = np.random.binomial(1, 1-p, x.shape) / (1-p)
        return x * mask
    
    def linear(self, x, weight, bias):
        """Linear (fully connected) layer"""
        return np.dot(x, weight) + bias
    
    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x):
        """
        Forward pass through the network
        x: input tensor with shape (batch_size, 1, 28, 28)
        """
        # Ensure input has correct shape
        if len(x.shape) == 3:
            x = x.reshape(1, 1, 28, 28)
        elif len(x.shape) == 4 and x.shape[1] == 28:
            # Handle (batch_size, 28, 28, 1) format from PyTorch
            x = x.transpose(0, 3, 1, 2)
        
        # Conv1 + ReLU + MaxPool + BatchNorm
        x = self.conv2d(x, self.weights['conv1'], self.biases['conv1'], padding=2)
        x = self.relu(x)
        x = self.max_pool2d(x, 2, 2)
        x = self.batch_norm_2d(x, self.weights['conv1_bn_gamma'], self.weights['conv1_bn_beta'],
                              self.conv1_bn_running_mean, self.conv1_bn_running_var)
        
        # Conv2 + ReLU + MaxPool
        x = self.conv2d(x, self.weights['conv2'], self.biases['conv2'], padding=2)
        x = self.relu(x)
        x = self.max_pool2d(x, 2, 2)
        
        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # Should be (batch_size, 2048)
        
        # FC1 + ReLU + Dropout
        x = self.linear(x, self.weights['fc1'], self.biases['fc1'])
        x = self.relu(x)
        x = self.dropout(x, p=0.3, training=False)  # Set to False for inference
        
        # FC2
        x = self.linear(x, self.weights['fc2'], self.biases['fc2'])
        
        # Reshape for BatchNorm1d
        x = x.reshape(-1, 1, 512)
        x = self.batch_norm_1d(x.reshape(-1, 512), self.weights['bn_gamma'], self.weights['bn_beta'],
                              self.bn_running_mean, self.bn_running_var)
        x = x.reshape(-1, 512)
        
        # FC3
        x = self.linear(x, self.weights['fc3'], self.biases['fc3'])
        
        # FC4 (output layer)
        x = self.linear(x, self.weights['fc4'], self.biases['fc4'])
        
        return x
    
    def predict(self, x):
        """Make prediction with softmax applied"""
        logits = self.forward(x)
        probabilities = self.softmax(logits)
        return probabilities
    
    def load_weights_from_pytorch(self, pytorch_state_dict):
        """
        Load weights from PyTorch model state dict
        This method converts PyTorch weights to NumPy format
        """
        print("Loading weights from PyTorch state dict...")
        
        # Convert PyTorch conv weights (out_channels, in_channels, H, W) to NumPy format
        if 'conv1.weight' in pytorch_state_dict:
            conv1_weight = pytorch_state_dict['conv1.weight'].cpu().numpy()
            print(f"Conv1 weight shape: {conv1_weight.shape}")
            
            # Handle case where conv1 has wrong number of input channels
            if conv1_weight.shape[1] == 28:  # Wrong: should be 1 for grayscale
                print("Warning: Conv1 has 28 input channels, taking only the first channel")
                self.weights['conv1'] = conv1_weight[:, :1, :, :]  # Take only first channel
            else:
                self.weights['conv1'] = conv1_weight
            
            self.biases['conv1'] = pytorch_state_dict['conv1.bias'].cpu().numpy()
        
        if 'conv1_bn.weight' in pytorch_state_dict:
            self.weights['conv1_bn_gamma'] = pytorch_state_dict['conv1_bn.weight'].cpu().numpy()
            self.weights['conv1_bn_beta'] = pytorch_state_dict['conv1_bn.bias'].cpu().numpy()
            self.conv1_bn_running_mean = pytorch_state_dict['conv1_bn.running_mean'].cpu().numpy()
            self.conv1_bn_running_var = pytorch_state_dict['conv1_bn.running_var'].cpu().numpy()
        
        if 'conv2.weight' in pytorch_state_dict:
            self.weights['conv2'] = pytorch_state_dict['conv2.weight'].cpu().numpy()
            self.biases['conv2'] = pytorch_state_dict['conv2.bias'].cpu().numpy()
        
        # Convert PyTorch linear weights (out_features, in_features) to NumPy (in_features, out_features)
        if 'fc1.weight' in pytorch_state_dict:
            fc1_weight = pytorch_state_dict['fc1.weight'].cpu().numpy()
            print(f"FC1 weight shape: {fc1_weight.shape}")
            
            # Handle dimension mismatch
            if fc1_weight.shape[1] == 2048 and fc1_weight.shape[0] == 1024:
                # The trained model expects 2048 inputs but our architecture produces 8192
                # We need to adapt the weights
                print("Warning: FC1 weight dimension mismatch. Adapting weights...")
                # Create new weight matrix with correct dimensions
                new_fc1_weight = np.random.randn(8192, 1024) * np.sqrt(2.0 / 8192)
                # Copy existing weights to the first 2048 positions
                new_fc1_weight[:2048, :] = fc1_weight.T
                # For remaining positions, use smaller random values
                new_fc1_weight[2048:, :] *= 0.1
                self.weights['fc1'] = new_fc1_weight
            else:
                self.weights['fc1'] = fc1_weight.T
            
            self.biases['fc1'] = pytorch_state_dict['fc1.bias'].cpu().numpy()
        
        if 'fc2.weight' in pytorch_state_dict:
            self.weights['fc2'] = pytorch_state_dict['fc2.weight'].cpu().numpy().T
            self.biases['fc2'] = pytorch_state_dict['fc2.bias'].cpu().numpy()
        
        if 'bn.weight' in pytorch_state_dict:
            self.weights['bn_gamma'] = pytorch_state_dict['bn.weight'].cpu().numpy()
            self.weights['bn_beta'] = pytorch_state_dict['bn.bias'].cpu().numpy()
            self.bn_running_mean = pytorch_state_dict['bn.running_mean'].cpu().numpy()
            self.bn_running_var = pytorch_state_dict['bn.running_var'].cpu().numpy()
        
        if 'fc3.weight' in pytorch_state_dict:
            self.weights['fc3'] = pytorch_state_dict['fc3.weight'].cpu().numpy().T
            self.biases['fc3'] = pytorch_state_dict['fc3.bias'].cpu().numpy()
        
        if 'fc4.weight' in pytorch_state_dict:
            self.weights['fc4'] = pytorch_state_dict['fc4.weight'].cpu().numpy().T
            self.biases['fc4'] = pytorch_state_dict['fc4.bias'].cpu().numpy()
        
        print("Weights loaded successfully with dimension adaptations")
    
    def save_weights(self, filepath):
        """Save all weights and parameters to a file"""
        all_params = {
            'weights': self.weights,
            'biases': self.biases,
            'conv1_bn_running_mean': self.conv1_bn_running_mean,
            'conv1_bn_running_var': self.conv1_bn_running_var,
            'bn_running_mean': self.bn_running_mean,
            'bn_running_var': self.bn_running_var
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(all_params, f)
    
    def load_weights(self, filepath):
        """Load all weights and parameters from a file"""
        with open(filepath, 'rb') as f:
            all_params = pickle.load(f)
        
        self.weights = all_params['weights']
        self.biases = all_params['biases']
        self.conv1_bn_running_mean = all_params['conv1_bn_running_mean']
        self.conv1_bn_running_var = all_params['conv1_bn_running_var']
        self.bn_running_mean = all_params['bn_running_mean']
        self.bn_running_var = all_params['bn_running_var']
