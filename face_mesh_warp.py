import mediapipe as mp
import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_image(title, img):
    """Utility function to display images"""
    plt.figure(figsize=(10, 10))
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def get_landmarks(image):
    """Extract facial landmarks using MediaPipe"""
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            raise ValueError("No face detected in the image")
            
        # Extract landmarks
        height, width = image.shape[:2]
        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append([x, y])
            
        return np.array(landmarks)

def visualize_landmarks(image, landmarks):
    """Visualize landmarks on the image"""
    img_copy = image.copy()
    for point in landmarks:
        cv2.circle(img_copy, tuple(point), 2, (0, 255, 0), -1)
    return img_copy

def create_triangle_mesh(landmarks, image_shape):
    """Create Delaunay triangulation from landmarks"""
    rect = (0, 0, image_shape[1], image_shape[0])
    subdiv = cv2.Subdiv2D(rect)
    
    for point in landmarks:
        subdiv.insert(tuple(map(float, point)))
    
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    
    # Convert triangles to indices
    triangle_indices = []
    for triangle in triangles:
        pt1 = (triangle[0], triangle[1])
        pt2 = (triangle[2], triangle[3])
        pt3 = (triangle[4], triangle[5])
        
        idx1 = np.where((landmarks == pt1).all(axis=1))[0]
        idx2 = np.where((landmarks == pt2).all(axis=1))[0]
        idx3 = np.where((landmarks == pt3).all(axis=1))[0]
        
        if len(idx1) > 0 and len(idx2) > 0 and len(idx3) > 0:
            triangle_indices.append([idx1[0], idx2[0], idx3[0]])
            
    return np.array(triangle_indices)

def visualize_triangulation(image, landmarks, triangles):
    """Visualize triangulation on the image"""
    img_copy = image.copy()
    for triangle in triangles:
        pt1 = tuple(landmarks[triangle[0]])
        pt2 = tuple(landmarks[triangle[1]])
        pt3 = tuple(landmarks[triangle[2]])
        
        cv2.line(img_copy, pt1, pt2, (0, 255, 0), 1)
        cv2.line(img_copy, pt2, pt3, (0, 255, 0), 1)
        cv2.line(img_copy, pt3, pt1, (0, 255, 0), 1)
    
    return img_copy

def warp_triangle(src_points, dst_points, src_triangle, dst_triangle, src_img, dst_img):
    """Warp a triangular region from source to destination"""
    # Find bounding box of triangle
    rect1 = cv2.boundingRect(src_triangle)
    rect2 = cv2.boundingRect(dst_triangle)
    
    # Offset points by left top corner of the bounding box
    src_triangle_offset = []
    dst_triangle_offset = []
    
    for i in range(3):
        src_triangle_offset.append(((src_points[i][0] - rect1[0]), (src_points[i][1] - rect1[1])))
        dst_triangle_offset.append(((dst_points[i][0] - rect2[0]), (dst_points[i][1] - rect2[1])))
    
    # Get mask by filling triangle
    mask = np.zeros((rect2[3], rect2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_triangle_offset), (1.0, 1.0, 1.0))
    
    # Apply warpAffine
    size1 = (rect1[2], rect1[3])
    size2 = (rect2[2], rect2[3])
    
    src_img_rect = src_img[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    
    if size1[0] == 0 or size1[1] == 0 or size2[0] == 0 or size2[1] == 0:
        return
    
    warp_mat = cv2.getAffineTransform(np.float32(src_triangle_offset), np.float32(dst_triangle_offset))
    warped_triangle = cv2.warpAffine(src_img_rect, warp_mat, (size2[0], size2[1]), None, 
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    # Copy triangular region of the rectangular patch to the output image
    warped_triangle = warped_triangle * mask
    
    dst_img[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = \
        dst_img[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] * (1 - mask) + warped_triangle

def warp_face(source_img, target_img):
    """Main function to warp face from source to target"""
    # Get landmarks for both images
    source_landmarks = get_landmarks(source_img)
    target_landmarks = get_landmarks(target_img)
    
    # Show original images with landmarks
    show_image("Source image with landmarks", 
              visualize_landmarks(source_img, source_landmarks))
    show_image("Target image with landmarks", 
              visualize_landmarks(target_img, target_landmarks))
    
    # Create triangulation
    triangles = create_triangle_mesh(source_landmarks, source_img.shape)
    
    # Show triangulation
    show_image("Source triangulation", 
              visualize_triangulation(source_img, source_landmarks, triangles))
    show_image("Target triangulation", 
              visualize_triangulation(target_img, target_landmarks, triangles))
    
    # Create output image
    warped_img = np.zeros_like(target_img)
    
    # Warp each triangle
    for triangle in triangles:
        src_triangle = source_landmarks[triangle]
        dst_triangle = target_landmarks[triangle]
        
        warp_triangle(src_triangle, dst_triangle, 
                     src_triangle, dst_triangle,
                     source_img, warped_img)
    
    # Show final result
    show_image("Warped Result", warped_img)
    
    return warped_img

# Use the functions
source_path =  '0d384dbbcc121ca5049c423f81c26e6a.png' # Replace with your source image path
target_path = 'vSYYZ1.png'  # Replace with your target image path

source_img = cv2.imread(source_path)
target_img = cv2.imread(target_path)

# Resize images if they're too large
max_size = 800
for img in [source_img, target_img]:
    if max(img.shape) > max_size:
        scale = max_size / max(img.shape)
        img = cv2.resize(img, None, fx=scale, fy=scale)

# Perform the warping
warped_result = warp_face(source_img, target_img)