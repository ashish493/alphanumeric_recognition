import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .optimized_network import OptimizedCNN
from PIL import Image

class OptimizedModel:
    """
    Model wrapper for the optimized CNN architecture trained in the notebook
    """

    def __init__(self, model_weights: str, device: str):
        """
        Initialize the optimized model
        
        Args:
            model_weights: Path to the best_emnist_model.pth file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.net = OptimizedCNN(num_classes=47, dropout_rate=0.5)
        self.weights = model_weights
        self.device = torch.device('cuda:0' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        
        # We'll handle preprocessing steps individually in the predict method
        # for more control and better debugging
        
        self._initialize()

    def _initialize(self):
        """Load the trained weights"""
        try:
            # Load checkpoint
            if torch.cuda.is_available() and self.device.type == 'cuda':
                checkpoint = torch.load(self.weights)
            else:
                checkpoint = torch.load(self.weights, map_location='cpu')
            
            # Extract model state dict from checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # Fallback for direct state dict
                state_dict = checkpoint
            
            # Load the state dict
            self.net.load_state_dict(state_dict)
            
            print(f"Successfully loaded optimized model weights from {self.weights}")
            if 'epoch' in checkpoint:
                print(f"Model was trained for {checkpoint['epoch']} epochs")
            if 'best_val_acc' in checkpoint:
                print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
                
        except FileNotFoundError:
            print(f"Error: Model weights file not found at {self.weights}")
            print("Please ensure the notebook training has completed and generated best_emnist_model.pth")
            return None
        except Exception as e:
            print(f"Error loading weights: {e}")
            return None
        
        # Set model to evaluation mode
        self.net.eval()
        
        # Move to specified device
        self.net.to(self.device)
        print(f"Model loaded on device: {self.device}")

    def predict(self, path, debug_visualization=False):
        """
        Predict the character from an image
        
        Args:
            path: Path to image file or BytesIO object
            debug_visualization: If True, save visualization images for debugging
            
        Returns:
            numpy array of probabilities for each class
        """
        try:
            # Open the image and convert to grayscale
            img = Image.open(path).convert('L')
            
            # Debug: Save original image if requested
            if debug_visualization:
                img.save('debug_original.png')
                print(f"Debug: Original image saved as debug_original.png (size: {img.size})")
            
            # EMNIST SPECIFIC PROCESSING:
            # The EMNIST dataset has a complex orientation pattern.
            # After multiple experiments, we've determined the correct transformation sequence.
            
            # Critical fix: rotate the input 270 degrees (or -90 degrees)
            # This matches the EMNIST training data orientation
            img = img.rotate(270, resample=Image.BICUBIC, expand=True)
            
            if debug_visualization:
                img.save('debug_rotated270.png')
                print(f"Debug: Image rotated 270° saved as debug_rotated270.png (size: {img.size})")
                
                # Save additional rotations for debugging
                img_copy = img.copy()
                img_copy = img_copy.transpose(Image.FLIP_LEFT_RIGHT)
                img_copy.save('debug_rotated270_flipped_h.png')
                
                img_copy = img.copy()
                img_copy = img_copy.transpose(Image.FLIP_TOP_BOTTOM)
                img_copy.save('debug_rotated270_flipped_v.png')
            
            # Final transformations based on testing
            # Keep only the 270° rotation for now
            
            if debug_visualization:
                img.save('debug_final_transform.png')
                print(f"Debug: Final transformed image saved as debug_final_transform.png (size: {img.size})")
            
            # Apply preprocessing
            with torch.no_grad():
                # Resize image to 28x28 (EMNIST size)
                img = img.resize((28, 28), Image.BICUBIC)
                
                if debug_visualization:
                    img.save('debug_resized.png')
                    print(f"Debug: Resized image saved as debug_resized.png")
                
                # Convert to tensor
                img_tensor = transforms.ToTensor()(img)
                
                # Invert colors to match training data (white background -> black, black text -> white)
                img_tensor = 1 - img_tensor
                
                # Apply normalization
                img_tensor = transforms.Normalize((0.1736,), (0.3317,))(img_tensor)
                
                # Debug: Save preprocessed image if requested
                if debug_visualization:
                    # Denormalize for visualization
                    img_preprocessed = (img_tensor.squeeze().numpy() * 0.3317) + 0.1736
                    img_preprocessed = np.clip(img_preprocessed * 255, 0, 255).astype('uint8')
                    Image.fromarray(img_preprocessed, mode='L').save('debug_preprocessed.png')
                    print(f"Debug: Preprocessed image saved as debug_preprocessed.png")
                
                # Debug: Save final model input image if requested
                if debug_visualization:
                    # Denormalize for visualization
                    img_final = (img_tensor.squeeze().numpy() * 0.3317) + 0.1736
                    img_final = np.clip(img_final * 255, 0, 255).astype('uint8')
                    Image.fromarray(img_final, mode='L').save('debug_model_input.png')
                    print(f"Debug: Final model input saved as debug_model_input.png")
                    print(f"Debug: Tensor stats - min: {img_tensor.min():.3f}, max: {img_tensor.max():.3f}, mean: {img_tensor.mean():.3f}")
                    
                    # Save a side-by-side comparison for easier debugging
                    try:
                        from PIL import ImageDraw
                        
                        # Create a side-by-side comparison
                        comparison = Image.new('L', (28*4, 28), color=128)
                        comparison.paste(Image.open('debug_original.png').resize((28, 28)), (0, 0))
                        comparison.paste(Image.open('debug_rotated270.png').resize((28, 28)), (28, 0))
                        comparison.paste(Image.open('debug_final_transform.png').resize((28, 28)), (56, 0))
                        comparison.paste(Image.fromarray(img_final, mode='L'), (84, 0))
                        
                        # Add labels
                        comparison = comparison.convert('RGB')
                        draw = ImageDraw.Draw(comparison)
                        draw.text((2, 2), "Orig", fill=(255, 0, 0))
                        draw.text((30, 2), "R270", fill=(255, 0, 0))
                        draw.text((58, 2), "Trans", fill=(255, 0, 0))
                        draw.text((86, 2), "Final", fill=(255, 0, 0))
                        
                        comparison.save('debug_comparison.png')
                        print(f"Debug: Processing steps comparison saved as debug_comparison.png")
                    except Exception as e:
                        print(f"Warning: Could not create comparison image: {e}")
                
                # Add batch dimension and move to device
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                # Run inference
                logits = self.net(img_tensor)
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(logits, dim=1)
                
                return probabilities[0].cpu().numpy()
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return uniform probabilities as fallback
            return np.ones(47) / 47

    def get_model_info(self):
        """Get information about the loaded model"""
        try:
            checkpoint = torch.load(self.weights, map_location='cpu')
            info = {
                'model_type': 'OptimizedCNN',
                'num_classes': 47,
                'architecture': 'CNN with 4 conv blocks + global avg pooling + 3 FC layers'
            }
            
            if 'epoch' in checkpoint:
                info['training_epochs'] = checkpoint['epoch']
            if 'best_val_acc' in checkpoint:
                info['best_validation_accuracy'] = f"{checkpoint['best_val_acc']:.2f}%"
            
            return info
        except:
            return {'error': 'Could not load model information'}
