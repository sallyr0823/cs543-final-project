import cv2
import sys
import numpy as np
import face_detection_pipeline as FacePipeline
from alpha_blending import FaceBlender
import shadow_recover 

class MakeupTransfer:
    """
    A class to handle makeup transfer between source and reference images.
    """
    def __init__(self, image_height=500):
        self.image_height = image_height
        self.face_blender = FaceBlender()

    def load_and_validate_images(self, source_path: str, makeup_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load and validate source and makeup reference images.
        
        Args:
            source_path (str): Path to source image
            makeup_path (str): Path to makeup reference image
            
        Returns:
            tuple[np.ndarray, np.ndarray]: Resized source and makeup images
        """
        source_image = cv2.imread(source_path)
        makeup_image = cv2.imread(makeup_path)
        
        if source_image is None:
            raise FileNotFoundError("Source image doesn't exist")
        if makeup_image is None:
            raise FileNotFoundError("Reference image doesn't exist")
            
        # First resize height
        source_image = self._resize_image(source_image)
        makeup_image = self._resize_image(makeup_image)
        
        # Then ensure same width by resizing makeup image to match source width
        makeup_image = cv2.resize(makeup_image, (source_image.shape[1], source_image.shape[0]))
        
        return source_image, makeup_image
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to maintain aspect ratio with target height."""
        ratio = self.image_height / image.shape[0]
        new_width = int(image.shape[1] * ratio)
        return cv2.resize(image, (new_width, self.image_height), 
                         interpolation=cv2.INTER_AREA)
    
    def process_image(self, image: np.ndarray) -> tuple:
        """Process image through feature detection and color space conversion."""
        l = FacePipeline.FaceLandmarkDetector()
        output_image, feature_points, triangulation,mask = (
            l.landmark_detection(image)
        )
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness = lab_image[:, :, 0]
        color = lab_image[:, :, 1:3]
        structure_layer, texture_layer = FacePipeline.wls_filter(lightness,mask)
        
        return (output_image, feature_points, triangulation,mask, 
                lightness, color, structure_layer, texture_layer)
    
    def transfer_makeup(self, source_image: np.ndarray, makeup_image: np.ndarray) -> np.ndarray:
        """
        Transfer makeup from reference image to source image.
        
        Args:
            source_image (np.ndarray): Source image
            makeup_image (np.ndarray): Makeup reference image
            
        Returns:
            np.ndarray: Result image with transferred makeup
        """
        # Process source and makeup images
        (source_output, source_features, source_tri,source_mask,
         source_light, source_color, source_struct, source_texture) = self.process_image(source_image)
        
        (makeup_output, makeup_features, makeup_tri,makeup_mask,
         makeup_light, makeup_color, makeup_struct, makeup_texture) = self.process_image(makeup_image)
        
        # Save source components
        cv2.imwrite('source_structure.png', (source_struct * 255).astype(np.uint8))
        cv2.imwrite('source_detail.png', (source_texture * 255).astype(np.uint8))
        
        # Convert and save source color (LAB color channels)
        source_color_vis = cv2.cvtColor(np.dstack([np.ones_like(source_color[:,:,0])*128, 
                                                source_color[:,:,0], 
                                                source_color[:,:,1]]), 
                                    cv2.COLOR_LAB2BGR)
        cv2.imwrite('source_color.png', source_color_vis)
        
        # Save makeup components
        cv2.imwrite('makeup_structure.png', (makeup_struct * 255).astype(np.uint8))
        cv2.imwrite('makeup_detail.png', (makeup_texture * 255).astype(np.uint8))
        
        # Convert and save makeup color (LAB color channels)
        makeup_color_vis = cv2.cvtColor(np.dstack([np.ones_like(makeup_color[:,:,0])*128, 
                                                makeup_color[:,:,0], 
                                                makeup_color[:,:,1]]), 
                                    cv2.COLOR_LAB2BGR)
        cv2.imwrite('makeup_color.png', makeup_color_vis)
        
        # Resize makeup components to match source dimensions if needed
        # if source_mask.shape != makeup_mask.shape:
        #     makeup_mask = cv2.resize(makeup_mask, 
        #                             (source_mask.shape[1], source_mask.shape[0]),
        #                             interpolation=cv2.INTER_NEAREST)
        
        if source_texture.shape != makeup_texture.shape:
            makeup_texture = cv2.resize(makeup_texture, 
                                      (source_texture.shape[1], source_texture.shape[0]),
                                      interpolation=cv2.INTER_AREA)
            
        if source_color.shape != makeup_color.shape:
            makeup_color = cv2.resize(makeup_color,
                                    (source_color.shape[1], source_color.shape[0]),
                                    interpolation=cv2.INTER_AREA)
            
        print("splitted")
        
        # Create tuples for blending
        source_tuple = (source_output, source_features, source_tri,source_mask)
        makeup_tuple = (makeup_output, makeup_features, makeup_tri,makeup_mask)
        
        print("tuples")
        
        print(f"Source shape: {source_image.shape}")
        print(f"Makeup shape: {makeup_image.shape}")
        print(f"Source mask shape: {source_mask.shape}")
        print(f"Makeup mask shape: {makeup_mask.shape}")
        
        # Initialize result image
        result_image = np.zeros_like(cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB))
        
        # Transfer skin detail
        result_skin_detail = self.face_blender.alpha_blend_images(
            source_texture, makeup_texture, 1, 1, source_tuple, makeup_tuple,True
        )
        
        print("transfered skin details")
        
        # Transfer color
        alpha_factor = 0.8
        result_image[:, :, 1:3] = self.face_blender.alpha_blend_images(
            source_color, makeup_color, 1 - alpha_factor, alpha_factor,
            source_tuple, makeup_tuple,True
        )
        
        print("color transfered")
        
        # Process structure layers
        scale_factor = 0.5
        source_struct_scaled = self._process_structure_layer(
            source_struct, scale_factor
        )
        makeup_struct_laplacian = cv2.Laplacian(makeup_struct, cv2.CV_64F)
        
        print("structure prepare")
        
        # Blend structure layers
        result_struct = self.face_blender.alpha_blend_images(
            source_struct_scaled, makeup_struct_laplacian, 1, 1,
            source_tuple, makeup_tuple,True
        )
        
        print("structure processed")
        
        # Combine detail and structure
        combined_layers = result_skin_detail + result_struct
        combined_layers = (combined_layers - np.min(combined_layers)) / (
            np.max(combined_layers) - np.min(combined_layers)
        )
        
        print("combined")
        
        # Set final lightness channel
        result_image[:, :, 0] = (combined_layers * 255).astype(np.uint8)
        
        print("final")
        
        return cv2.cvtColor(result_image, cv2.COLOR_LAB2BGR)
    
    def _process_structure_layer(self, layer: np.ndarray, scale_factor: float) -> np.ndarray:
        """Process structure layer with scaling."""
        height, width = layer.shape
        scaled_size = (int(width * scale_factor), int(height * scale_factor))
        halved = cv2.resize(layer, scaled_size, interpolation=cv2.INTER_AREA)
        return cv2.resize(halved, (width, height), interpolation=cv2.INTER_AREA)

def main():
    """Main function to run the makeup transfer application."""
    if len(sys.argv) != 3:
        print("Syntax Error: Proper Syntax = python3 Makeup.py source reference")
        sys.exit(1)
        
    try:
        makeup_transfer = MakeupTransfer()
        source_image, makeup_image = makeup_transfer.load_and_validate_images(
            sys.argv[1], sys.argv[2]
        )
        
        result_image = makeup_transfer.transfer_makeup(source_image, makeup_image)
        
        # Display results
        comparison = np.hstack((source_image, result_image, makeup_image))
        cv2.imshow("Output", comparison)
        
        # Apply shade recovery
        shadow_recover.shadeRecover(source_image, result_image, makeup_image)
        
        # Wait for key press and save if 's' is pressed
        key = cv2.waitKey(0)
        if key & 0xFF == ord('s'):
            cv2.imwrite('out.jpg', comparison)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()