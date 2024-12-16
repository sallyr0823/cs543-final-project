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
        
        x, y, w, h = cv2.boundingRect(points[np.unique(triangles.flatten())])
        masked_face = cv2.bitwise_and(image, image, mask=mask)
        cropped_face = masked_face[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]
        
        b, g, r = cv2.split(cropped_face)
        cropped_face_rgba = cv2.merge((b, g, r, cropped_mask))
        
        return cropped_face_rgba

    @staticmethod
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
    def replace_face_in_fullbody(fullbody_image, makeup_face, face_coords):
        """Replace face in fullbody image with warped makeup face."""
        x1, y1, x2, y2 = face_coords
        
        # Get landmarks and mesh for makeup face
        landmark_detector = FaceLandmarkDetector()
        makeup_landmarks = landmark_detector.get_landmarks(makeup_face)
        triangles = FaceWarper.create_triangle_mesh(makeup_landmarks, makeup_face.shape)
        cropped_makeup_face = FaceWarper.extract_face_region(makeup_face, makeup_landmarks, triangles)
        cv2.imwrite('cropped_makeup_face.png', cropped_makeup_face)
        
        # Resize makeup face to match target face dimensions
        face_height = y2 - y1
        face_width = x2 - x1
        makeup_face_resized = cv2.resize(cropped_makeup_face, (face_width, face_height))
        
        # Create result image and blending mask
        result_image = fullbody_image.copy()
        mask = np.zeros((face_height, face_width), dtype=np.float32)
        center = (face_width // 2, face_height // 2)
        radius = min(face_width, face_height) // 2
        cv2.circle(mask, center, radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (19, 19), 0)
        
        # Blend faces
        for c in range(3):
            result_image[y1:y2, x1:x2, c] = (
                makeup_face_resized[:, :, c] * mask +
                fullbody_image[y1:y2, x1:x2, c] * (1 - mask)
            )
        
        return result_image

class FaceMask:
    FACIAL_PARTS = {
        "Face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
        "Left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        "Right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        "Lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
        "Left_eyebrow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
        "Right_eyebrow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
        "Nose": [8, 190, 128, 98, 327, 326, 327, 2, 327, 417, 8]
    }

    @staticmethod
    def create_mask(image, landmarks):
        """Create binary mask of facial features."""
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        for part_name, indices in FaceMask.FACIAL_PARTS.items():
            points = np.array([landmarks[idx] for idx in indices], dtype=np.int32)
            
            if part_name in ["Face_oval", "Nose"]:
                cv2.polylines(mask, [points], True, 0, 2)
            else:
                cv2.fillPoly(mask, [points], 0)
                
                if part_name in ["Left_eye", "Right_eye", "Lips"]:
                    center = np.mean(points, axis=0)
                    points_centered = points - center
                    points_inner = (points_centered * 0.8 + center).astype(np.int32)
                    cv2.fillPoly(mask, [points_inner], 0)
        
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

def main():
    # Initialize components
    image_utils = ImageUtils()
    face_detector = FaceDetector()
    landmark_detector = FaceLandmarkDetector()
    
    # Load images
    current_dir = os.getcwd()
    source_path = os.path.join(current_dir, "0d384dbbcc121ca5049c423f81c26e6a.png")
    target_path = os.path.join(current_dir, "fullbody_model.png")
    
    if not os.path.exists(source_path) or not os.path.exists(target_path):
        raise FileNotFoundError("Source or target image not found")
    
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
    extracted_source_face = FaceWarper.extract_face_region(source_img, source_landmarks, source_triangles)
    extracted_target_face = FaceWarper.extract_face_region(target_img, target_landmarks, target_triangles)
    
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
    
    # Replace face in fullbody image
    face_coords = face_detector.detect_face(target_img)
    final_result = FaceWarper.replace_face_in_fullbody(target_img, warped_img, face_coords)
    
    # Save final result
    cv2.imwrite('final_result.png', final_result)
    
    # Display results
    image_utils.show_image('Extracted Source Face', extracted_source_face)
    image_utils.show_image('Extracted Target Face', extracted_target_face)

if __name__ == "__main__":
    main()
