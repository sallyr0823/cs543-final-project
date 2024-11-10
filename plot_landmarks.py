import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_face_mesh_points(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    
    # Initialize MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
        
        # Read image and convert to RGB
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            print("No face detected")
            return
            
        # Get landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Create figure with larger size
        plt.figure(figsize=(15, 15))
        
        # Plot the image
        plt.imshow(image_rgb)
        
        # Get height and width of image
        height, width = image.shape[:2]
        
        # Plot each landmark with its index
        for idx, landmark in enumerate(face_landmarks.landmark):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            
            # Plot point
            plt.plot(x, y, 'r.', markersize=1)
            
            # Add index number (showing every 10th number to avoid clutter)
            if idx % 10 == 0:  # Change this number to show more or fewer labels
                plt.text(x+2, y+2, str(idx), fontsize=8, color='white',
                        bbox=dict(facecolor='red', alpha=0.7, edgecolor='none', pad=1))
        
        plt.axis('off')
        plt.savefig('face_mesh_points.jpg', bbox_inches='tight', dpi=300)
        plt.close()

        # Create an additional visualization with different regions colored differently
        image_copy = image_rgb.copy()
        
        # Define different regions with their point indices
        regions = {
            'Face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            'Left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'Right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'Lips': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
            'Left_eyebrow': [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
            'Right_eyebrow': [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
            'Nose': [168, 193, 245, 188, 174, 217, 126, 142, 97, 98, 129, 49, 131, 134, 236, 239, 242, 183, 244]
        }
        
        # Create a new figure for regions visualization
        plt.figure(figsize=(15, 15))
        
        # Plot base image
        plt.imshow(image_rgb)
        
        # Plot each region with different colors
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'white']
        for (region, points), color in zip(regions.items(), colors):
            # Plot points for this region
            for point_idx in points:
                landmark = face_landmarks.landmark[point_idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                plt.plot(x, y, '.', color=color, markersize=5)
            
            # Add legend entry
            plt.plot([], [], '.', color=color, label=region)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('off')
        plt.savefig('face_mesh_regions.jpg', bbox_inches='tight', dpi=300)
        plt.close()

# Use the function
image_path = 'vSYYZ1.png'  # Replace with your image path
plot_face_mesh_points(image_path)