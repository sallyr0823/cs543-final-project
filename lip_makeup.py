import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def gaussian(x, sigma):
    """Gaussian function as described in the paper"""
    return np.exp(-x**2 / (2*sigma**2))

def apply_lip_makeup(source_img, example_img, source_mask, example_mask):
    # Convert to LAB color space
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
    example_lab = cv2.cvtColor(example_img, cv2.COLOR_BGR2LAB)
    
    # Separate L, a, b channels
    source_l = source_lab[:,:,0].astype(float)
    example_l = example_lab[:,:,0].astype(float)
    
    # Histogram equalization for L channel as mentioned in paper
    source_l_eq = cv2.equalizeHist(source_lab[:,:,0])
    example_l_eq = cv2.equalizeHist(example_lab[:,:,0])
    
    # Create output array
    makeup_result = np.copy(source_lab)
    
    # Get lip region coordinates
    source_lip_coords = np.where(source_mask == 2)  # 2 denotes lip region
    example_lip_coords = np.where(example_mask == 2)
    
    # Parameters for Gaussian functions
    sigma_spatial = 5.0  # For spatial distance
    sigma_intensity = 5.0  # For intensity difference
    
    # For each pixel in source lip region
    for y, x in zip(*source_lip_coords):
        # Get source intensity value
        I_p = source_l_eq[y, x]
        
        # Compute distances and intensity differences for all example lip pixels
        spatial_dists = np.sqrt(
            (example_lip_coords[0] - y)**2 + 
            (example_lip_coords[1] - x)**2
        )
        intensity_diffs = np.abs(example_l_eq[example_lip_coords] - I_p)
        
        # Compute Gaussian weights
        spatial_weights = gaussian(spatial_dists, sigma_spatial)
        intensity_weights = gaussian(intensity_diffs, sigma_intensity)
        total_weights = spatial_weights * intensity_weights
        
        # Find best matching pixel
        best_match_idx = np.argmax(total_weights)
        q_y, q_x = example_lip_coords[0][best_match_idx], example_lip_coords[1][best_match_idx]
        
        # Transfer the color (all LAB channels)
        makeup_result[y, x] = example_lab[q_y, q_x]
    
    # Gradient domain fusion for L channel
    makeup_result[:,:,0] = gradient_fusion(
        source_lab[:,:,0], 
        makeup_result[:,:,0], 
        source_mask == 2
    )
    
    # Convert back to BGR
    return cv2.cvtColor(makeup_result, cv2.COLOR_LAB2BGR)

def gradient_fusion(source, target, mask):
    """
    Perform gradient domain fusion for L channel as mentioned in paper
    """
    # Compute gradients
    source_grad_x = cv2.Sobel(source, cv2.CV_64F, 1, 0, ksize=3)
    source_grad_y = cv2.Sobel(source, cv2.CV_64F, 0, 1, ksize=3)
    target_grad_x = cv2.Sobel(target, cv2.CV_64F, 1, 0, ksize=3)
    target_grad_y = cv2.Sobel(target, cv2.CV_64F, 0, 1, ksize=3)
    
    # Use target gradients in lip region
    grad_x = np.where(mask, target_grad_x, source_grad_x)
    grad_y = np.where(mask, target_grad_y, source_grad_y)
    
    # Solve Poisson equation
    fused = np.copy(source)
    mask_float = mask.astype(float)
    
    # Simple iterative solver (could be replaced with more efficient solver)
    for _ in range(50):
        laplacian = cv2.Laplacian(fused, cv2.CV_64F)
        div = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3) + \
              cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
        fused += 0.1 * (div - laplacian) * mask_float
    
    return np.clip(fused, 0, 255).astype(np.uint8)