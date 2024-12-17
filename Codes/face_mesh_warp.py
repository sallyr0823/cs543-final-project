import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mtcnn import MTCNN

class ImageUtils:
    @staticmethod
    def show_image(title, img):
        """Display images using matplotlib."""
        plt.figure(figsize=(10, 10))
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()

    @staticmethod
    def resize_image(img, max_size=800):
        """Resize image if it exceeds maximum size while maintaining aspect ratio."""
        if max(img.shape) > max_size:
            scale = max_size / max(img.shape)
            return cv2.resize(img, None, fx=scale, fy=scale)
        return img

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_face(self, image):
        """Detect face in image using MTCNN."""
        results = self.detector.detect_faces(image)
        if not results:
            raise ValueError("No face detected in the image")
        print('MTCC perform face detection!!!')
        x1, y1, width, height = results[0]["box"]
        return x1, y1, x1 + width, y1 + height

class FaceLandmarkDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detector = FaceDetector()

    def get_landmarks(self, image):
        """Extract facial landmarks using MediaPipe Face Mesh."""
        try:
            return self._process_image(image)
        except ValueError:
            # If direct detection fails, try with face detection first
            x1, y1, x2, y2 = self.face_detector.detect_face(image)
            face_image = image[y1:y2, x1:x2]
            landmarks = self._process_image(face_image)
            # Adjust coordinates back to original image
            return np.array([[x + x1, y + y1] for x, y in landmarks])

    def _process_image(self, image):
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if not results.multi_face_landmarks:
                raise ValueError("No face landmarks detected")
            
            height, width = image.shape[:2]
            return np.array([
                [int(landmark.x * width), int(landmark.y * height)]
                for landmark in results.multi_face_landmarks[0].landmark
            ])

class FaceWarper:
    @staticmethod
    def create_triangle_mesh(landmarks, image_shape):
        """Create Delaunay triangulation from landmarks."""
        rect = (0, 0, image_shape[1], image_shape[0])
        subdiv = cv2.Subdiv2D(rect)
        
        for point in landmarks:
            subdiv.insert(tuple(map(float, point)))
            
        triangles = subdiv.getTriangleList()
        return FaceWarper._convert_triangles_to_indices(triangles, landmarks)

    @staticmethod
    def _convert_triangles_to_indices(triangles, landmarks):
        triangle_indices = []
        for t in triangles:
            pt1, pt2, pt3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])
            idx1 = np.where((landmarks == pt1).all(axis=1))[0]
            idx2 = np.where((landmarks == pt2).all(axis=1))[0]
            idx3 = np.where((landmarks == pt3).all(axis=1))[0]
            
            if len(idx1) > 0 and len(idx2) > 0 and len(idx3) > 0:
                triangle_indices.append([idx1[0], idx2[0], idx3[0]])
                
        return np.array(triangle_indices)

    @staticmethod
    def extract_face_region(image, landmarks, triangles):
        """Extract face region using landmarks and triangulation."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points = np.array(landmarks, dtype=np.int32)
        
        for triangle in triangles:
            pts = points[triangle]
            cv2.fillConvexPoly(mask, pts, 255)
            
        # Save the full face mask before cropping
        cv2.imwrite('full_face_mask.png', mask)
        
        x, y, w, h = cv2.boundingRect(points[np.unique(triangles.flatten())])
        masked_face = cv2.bitwise_and(image, image, mask=mask)
        cropped_face = masked_face[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        
        b, g, r = cv2.split(cropped_face)
        cropped_face_rgba = cv2.merge((b, g, r, cropped_mask))
        
        # Return both the face and its coordinates and full face mask
        return cropped_face_rgba, (x, y, w, h), mask

    @staticmethod
    def warp_triangle(src_points, dst_points, src_triangle, dst_triangle, src_img, dst_img):
        """Warp a triangular region from source to destination."""
        rect1 = cv2.boundingRect(src_triangle)
        rect2 = cv2.boundingRect(dst_triangle)
        
        src_triangle_offset = [(p[0] - rect1[0], p[1] - rect1[1]) for p in src_points]
        dst_triangle_offset = [(p[0] - rect2[0], p[1] - rect2[1]) for p in dst_points]
        
        # Create mask
        mask = np.zeros((rect2[3], rect2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(dst_triangle_offset), (1.0, 1.0, 1.0))
        
        size1 = (rect1[2], rect1[3])
        size2 = (rect2[2], rect2[3])
        
        if 0 in size1 or 0 in size2:
            return
            
        # Convert source image region to float32
        src_img_rect = src_img[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]].astype(np.float32)
        
        # Get the affine transform
        warp_mat = cv2.getAffineTransform(
            np.float32(src_triangle_offset),
            np.float32(dst_triangle_offset)
        )
        
        # Warp the triangle
        warped_triangle = cv2.warpAffine(
            src_img_rect,
            warp_mat,
            (size2[0], size2[1]),
            None,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        # Apply mask to warped triangle
        warped_triangle = warped_triangle * mask
        
        # Get the destination region
        dst_region = dst_img[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]].astype(np.float32)
        
        # Blend the triangles
        blended = dst_region * (1 - mask) + warped_triangle
        
        # Convert back to uint8 before assigning to destination image
        dst_img[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]] = blended.astype(np.uint8)

    @staticmethod
    def replace_face_in_fullbody(fullbody_image, warped_img, face_mask):
        """Replace face in fullbody image with warped makeup face using the original face mask."""
        result_image = fullbody_image.copy()
        
        # Convert mask to float32 and normalize
        mask = face_mask.astype(np.float32) / 255.0
        
        # Apply Gaussian blur to smooth the edges
        mask = cv2.GaussianBlur(mask, (19, 19), 0)
        
        # Expand mask to 3 channels
        mask = np.stack([mask] * 3, axis=-1)
        
        # Blend using the mask
        result_image = warped_img * mask + fullbody_image * (1 - mask)
        
        return result_image.astype(np.uint8)



def smooth_mask(mask, height, width):
    """
    Apply specialized smoothing function from the paper:
    β(p) = min(1 - k(q) * exp(-(q-p)²/(2σ²)))
    """
    # Calculate σ²
    sigma_squared = min(height, width) / 25.0
    
    # Create output mask
    smoothed_mask = np.zeros_like(mask, dtype=np.float32)
    
    # Define k(q) values
    def get_k_value(pixel_value):
        if abs(pixel_value - 0.3) < 0.01:  # Eyebrow region (0.3)
            return 0.7
        elif abs(pixel_value - 1.0) < 0.01:  # Skin area (1.0)
            return 0.0
        else:  # Other facial components (0.0)
            return 1.0

    # For each pixel p
    for y in range(height):
        for x in range(width):
            min_value = float('inf')
            p = np.array([x, y])
            
            # Consider a window around the pixel for efficiency
            window_size = int(3 * np.sqrt(sigma_squared))  # 3σ captures most of the effect
            y_start = max(0, y - window_size)
            y_end = min(height, y + window_size + 1)
            x_start = max(0, x - window_size)
            x_end = min(width, x + window_size + 1)
            
            # For each pixel q in window
            for qy in range(y_start, y_end):
                for qx in range(x_start, x_end):
                    q = np.array([qx, qy])
                    
                    # Calculate squared distance
                    dist_squared = np.sum((p - q) ** 2)
                    
                    # Get k(q) based on the region
                    k = get_k_value(mask[qy, qx])
                    
                    # Calculate the smoothing function
                    value = 1.0 - k * np.exp(-dist_squared / (2 * sigma_squared))
                    min_value = min(min_value, value)
            
            smoothed_mask[y, x] = min_value

    return smoothed_mask

class FaceMask:
    FACIAL_PARTS = {
        "Face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
        "Left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        "Right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        "Lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
        "Left_eyebrow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
        "Right_eyebrow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
        # "Nose": [8, 190, 128, 98, 327, 326, 327, 2, 327, 417, 8]
        # Modified nose points - removing top points
        "Nose": [190, 128, 98, 327, 326, 327, 2, 327, 417]  
    }

    @staticmethod
    def create_mask(image, landmarks):
        """Create binary mask of facial features with specific fill rules."""
        height, width = image.shape[:2]
        
        # Start with black background
        mask = np.zeros((height, width), dtype=np.float32)
        
        # 1. Find and fill face oval with white
        face_points = np.array([landmarks[idx] for idx in FaceMask.FACIAL_PARTS["Face_oval"]], dtype=np.int32)
        cv2.fillPoly(mask, [face_points], 1.0)  # Fill with white (1.0)
        
        # 2. Fill eyes with black
        for eye_part in ["Left_eye", "Right_eye"]:
            eye_points = np.array([landmarks[idx] for idx in FaceMask.FACIAL_PARTS[eye_part]], dtype=np.int32)
            cv2.fillPoly(mask, [eye_points], 0.0)  # Fill with black (0.0)
        
        # 3. Fill mouth with black
        mouth_points = np.array([landmarks[idx] for idx in FaceMask.FACIAL_PARTS["Lips"]], dtype=np.int32)
        cv2.fillPoly(mask, [mouth_points], 0.0)
        
        # 4. Fill eyebrows with gray (0.3)
        for brow_part in ["Left_eyebrow", "Right_eyebrow"]:
            brow_points = np.array([landmarks[idx] for idx in FaceMask.FACIAL_PARTS[brow_part]], dtype=np.int32)
            cv2.fillPoly(mask, [brow_points], 0.3)
        
       # Modified nose contour - only lower part
        nose_points = np.array([landmarks[idx] for idx in FaceMask.FACIAL_PARTS["Nose"]], dtype=np.int32)
        cv2.polylines(mask, [nose_points], False, 0.0, 2)  # Changed to False to not close the contour
        
        # Apply the specialized smoothing
        smoothed_mask = smooth_mask(mask, height, width)
        
        # Convert to proper format (0-255 range)
        smoothed_mask = (smoothed_mask * 255).astype(np.uint8)
        
        return smoothed_mask

def main():
    # Initialize components
    image_utils = ImageUtils()
    face_detector = FaceDetector()
    landmark_detector = FaceLandmarkDetector()
    
    # Load images
    current_dir = os.getcwd()
    source_path = "/Users/sallyr/Documents/cs534/final-project/Digital_MakeUp_Face_Generation/SampleImages/withoutMakeup.jpg"
    target_path = "/Users/sallyr/Documents/cs534/final-project/Digital_MakeUp_Face_Generation/SampleImages/Makeup.jpg"
   
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    
    # Resize images if needed
    source_img = image_utils.resize_image(source_img)
    target_img = image_utils.resize_image(target_img)
    
    # Get landmarks
    source_landmarks = landmark_detector.get_landmarks(source_img)
    target_landmarks = landmark_detector.get_landmarks(target_img)
    
    # Create triangle meshes
    source_triangles = FaceWarper.create_triangle_mesh(source_landmarks, source_img.shape)
    target_triangles = FaceWarper.create_triangle_mesh(target_landmarks, target_img.shape)
    
    # Extract faces and save intermediate results
    extracted_source_face, source_coords, source_mask = FaceWarper.extract_face_region(source_img, source_landmarks, source_triangles)
    extracted_target_face, target_coords, full_target_mask = FaceWarper.extract_face_region(target_img, target_landmarks, target_triangles)
    
    # Save coordinates to a file
    with open('face_coordinates.txt', 'w') as f:
        f.write(f"Source face coordinates (x, y, w, h): {source_coords}\n")
        f.write(f"Target face coordinates (x, y, w, h): {target_coords}\n")
    
    cv2.imwrite('extracted_source_face.png', extracted_source_face)
    cv2.imwrite('extracted_target_face.png', extracted_target_face)
    
    # Create and save facial mask for target face
    target_mask = FaceMask.create_mask(extracted_target_face, landmark_detector.get_landmarks(extracted_target_face))
    cv2.imwrite('facial_mask.png', target_mask)
    
    # Perform face warping
    warped_img = np.zeros_like(target_img)
    for triangle in source_triangles:
        src_triangle = source_landmarks[triangle]
        dst_triangle = target_landmarks[triangle]
        FaceWarper.warp_triangle(
            src_triangle,
            dst_triangle,
            src_triangle,
            dst_triangle,
            source_img,
            warped_img
        )
    
    # Save warped result
    cv2.imwrite('warped_result.png', warped_img)
    
    print("Warped image shape:", warped_img.shape)
    print("Warped image min/max values:", np.min(warped_img), np.max(warped_img))
    
    # Replace face in fullbody image using stored coordinates
    final_result = FaceWarper.replace_face_in_fullbody(target_img, warped_img, full_target_mask)
 
    # Save final result
    cv2.imwrite('final_result.png', final_result)
    
    # Display results
    image_utils.show_image('Extracted Source Face', extracted_source_face)
    image_utils.show_image('Extracted Target Face', extracted_target_face)

if __name__ == "__main__":
    main()