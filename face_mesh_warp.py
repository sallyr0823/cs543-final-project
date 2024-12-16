import os
import mediapipe as mp
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mtcnn import MTCNN


def show_image(title, img):
    """Utility function to display images"""
    plt.figure(figsize=(10, 10))
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


# face detection with mtcc
def detect_face_mtcnn(image):
    detector = MTCNN()
    results = detector.detect_faces(image)

    if not results:
        raise ValueError("No face detected in the image")

    print('MTCC perform face detection!!!')
    x1, y1, width, height = results[0]["box"]
    x2, y2 = x1 + width, y1 + height
    return x1, y1, x2, y2


def get_landmarks(image):
    """Extract facial landmarks using MediaPipe Face Mesh."""
    mp_face_mesh = mp.solutions.face_mesh

    # Initialize face detection with default full-image bounding box
    x1, y1, x2, y2 = 0, 0, image.shape[1], image.shape[0]
    try:
        # Try detecting landmarks directly without detecting face first
        face_image = image[y1:y2, x1:x2]

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:

            # Convert BGR to RGB
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(face_image_rgb)

            if not results.multi_face_landmarks:
                raise ValueError("No face landmarks detected in the face image")

            # Extract landmarks
            height, width = face_image.shape[:2]
            landmarks = []
            for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
                # Compute coordinates within the face image
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                # Adjust coordinates back to the original image
                x += x1
                y += y1
                landmarks.append([x, y])
            return np.array(landmarks)

    except ValueError:
        # If no landmarks detected, try detecting the face first
        x1, y1, x2, y2 = detect_face_mtcnn(image)

        # Retry landmark detection with cropped face region
        face_image = image[y1:y2, x1:x2]

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:

            # Convert BGR to RGB
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(face_image_rgb)

            if not results.multi_face_landmarks:
                raise ValueError(
                    "No face landmarks detected in the face image after face detection"
                )

            # Extract landmarks
            height, width = face_image.shape[:2]
            landmarks = []
            for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
                # Compute coordinates within the face image
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                # Adjust coordinates back to the original image
                x += x1
                y += y1
                landmarks.append([x, y])
        return np.array(landmarks)


def segment_face_parts(image, landmarks):
    """
    Segment facial parts (eyes, nose, mouth) from the image based on landmarks.

    Parameters:
    - image: The original image.
    - landmarks: An array of facial landmarks.

    Returns:
    - segments: A dictionary containing segmented facial parts as images.
    """
    # Define landmark indices for different facial parts
    FACIAL_PARTS = {
        # 'Face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
        "Left_eye": [
            33,
            7,
            163,
            144,
            145,
            153,
            154,
            155,
            133,
            173,
            157,
            158,
            159,
            160,
            161,
            246,
        ],
        "Right_eye": [
            362,
            382,
            381,
            380,
            374,
            373,
            390,
            249,
            263,
            466,
            388,
            387,
            386,
            385,
            384,
            398,
        ],
        "Lips": [
            61,
            146,
            91,
            181,
            84,
            17,
            314,
            405,
            321,
            375,
            291,
            409,
            270,
            269,
            267,
            0,
            37,
            39,
            40,
            185,
        ],
        "Left_eyebrow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
        "Right_eyebrow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
        "Nose": [8, 190, 128, 98, 327, 326, 327, 2, 327, 417, 8],
    }

    segments = {}

    for part_name, indices in FACIAL_PARTS.items():
        # Get the points for the facial part
        points = np.array([landmarks[idx] for idx in indices], np.int32)

        # Create a mask for the facial part
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        # Extract the facial part using the mask
        segmented_part = cv2.bitwise_and(image, image, mask=mask)

        # Crop the segmented part to its bounding rectangle
        x, y, w, h = cv2.boundingRect(points)
        cropped_part = segmented_part[y : y + h, x : x + w]

        segments[part_name] = cropped_part

    return segments


def visualize_landmarks(image, landmarks):
    """Draw landmarks and their indices on the image."""
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
        src_triangle_offset.append(
            ((src_points[i][0] - rect1[0]), (src_points[i][1] - rect1[1]))
        )
        dst_triangle_offset.append(
            ((dst_points[i][0] - rect2[0]), (dst_points[i][1] - rect2[1]))
        )

    # Get mask by filling triangle
    mask = np.zeros((rect2[3], rect2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_triangle_offset), (1.0, 1.0, 1.0))

    # Apply warpAffine
    size1 = (rect1[2], rect1[3])
    size2 = (rect2[2], rect2[3])

    src_img_rect = src_img[
        rect1[1] : rect1[1] + rect1[3], rect1[0] : rect1[0] + rect1[2]
    ]

    if size1[0] == 0 or size1[1] == 0 or size2[0] == 0 or size2[1] == 0:
        return

    warp_mat = cv2.getAffineTransform(
        np.float32(src_triangle_offset), np.float32(dst_triangle_offset)
    )
    warped_triangle = cv2.warpAffine(
        src_img_rect,
        warp_mat,
        (size2[0], size2[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Copy triangular region of the rectangular patch to the output image
    warped_triangle = warped_triangle * mask

    dst_img[rect2[1] : rect2[1] + rect2[3], rect2[0] : rect2[0] + rect2[2]] = (
        dst_img[rect2[1] : rect2[1] + rect2[3], rect2[0] : rect2[0] + rect2[2]]
        * (1 - mask)
        + warped_triangle
    )


def extract_face_region(image, landmarks, triangles):
    """
    Extracts the face region from the image using landmarks and triangulation
    
    Args:
        image: Original image
        landmarks: Facial landmark points
        triangles: Triangulation indices defining the face mesh
    """
    # Create a mask using the triangulation
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Convert landmarks to numpy array if not already
    points = np.array(landmarks, dtype=np.int32)
    
    # Draw all triangles on the mask
    for triangle in triangles:
        pts = points[triangle]
        cv2.fillConvexPoly(mask, pts, 255)
    
    # Get the bounding rectangle of the triangulated region
    x, y, w, h = cv2.boundingRect(points[np.unique(triangles.flatten())])
    
    # Apply the mask to the image
    masked_face = cv2.bitwise_and(image, image, mask=mask)
    
    # Crop to the bounding rectangle
    cropped_face = masked_face[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    
    # Add alpha channel (transparency)
    b, g, r = cv2.split(cropped_face)
    cropped_face_rgba = cv2.merge((b, g, r, cropped_mask))
    
    show_image('extracted face',cropped_face)
    return cropped_face_rgba

def warp_face(source_img, target_img):
    """Main function to warp face from source to target"""
    # Get landmarks for both images
    source_landmarks = get_landmarks(source_img)
    target_landmarks = get_landmarks(target_img)

    # Show original images with landmarks
    show_image(
        "Source image with landmarks", visualize_landmarks(source_img, source_landmarks)
    )
    show_image(
        "Target image with landmarks", visualize_landmarks(target_img, target_landmarks)
    )

    # Create triangulation
    target_triangles = create_triangle_mesh(target_landmarks, target_img.shape)
    # extract the image from the target/dst image
    extracted_target_face=extract_face_region(target_img, target_landmarks, target_triangles)
    cv2.imwrite("extracted_target_face.png", extracted_target_face)

    source_triangles = create_triangle_mesh(source_landmarks,source_img.shape)
    extracted_source_face = extract_face_region(source_img,source_landmarks,source_triangles)
    cv2.imwrite('extracted_source_face.png',extracted_source_face)
    # Show triangulation
    show_image(
        "Source triangulation",
        visualize_triangulation(source_img, source_landmarks, source_triangles),
    )
    show_image(
        "Target triangulation",
        visualize_triangulation(target_img, target_landmarks, target_triangles),
    )

    # Create output image
    warped_img = np.zeros_like(target_img)

    # Warp each triangle
    for triangle in source_triangles:
        src_triangle = source_landmarks[triangle]
        dst_triangle = target_landmarks[triangle]

        warp_triangle(
            src_triangle,
            dst_triangle,
            src_triangle,
            dst_triangle,
            source_img,
            warped_img,
        )

    # Show final result
    show_image("Warped Result", warped_img)
    output_path = "warped_result.png" 
    cv2.imwrite(output_path, warped_img)
    return warped_img

def create_facial_mask(image, landmarks):
    """
    Creates a binary mask highlighting facial features using specific landmark indices
    
    Args:
        image: Original image
        landmarks: Array of facial landmark points
    
    Returns:
        mask: Binary mask image with facial features
    """
    # Create a black background
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    
    # Define facial part indices
    FACIAL_PARTS = {
        "Face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
        "Left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        "Right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        "Lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
        "Left_eyebrow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
        "Right_eyebrow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
        "Nose": [8, 190, 128, 98, 327, 326, 327, 2, 327, 417, 8]
    }

    # Draw each facial part
    for part_name, indices in FACIAL_PARTS.items():
        # Get the points for the facial part
        points = np.array([landmarks[idx] for idx in indices], dtype=np.int32)
        if part_name == "Face_oval" or part_name=="Nose":
        # Draw face outline as a line
            cv2.polylines(mask, [points], True, 0, 2)
        # Fill the facial part area
        else:
            cv2.fillPoly(mask, [points], 0)
        
        # If it's eyes or mouth, create inner area
        if part_name in ["Left_eye", "Right_eye", "Lips"]:
            # Create slightly smaller inner polygon
            center = np.mean(points, axis=0)
            points_centered = points - center
            points_scaled = points_centered * 0.8  # Scale down to 80%
            points_inner = (points_scaled + center).astype(np.int32)
            cv2.fillPoly(mask, [points_inner], 0)

    # Optional: Apply slight Gaussian blur for smoother edges
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Threshold to make it purely binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask

def save_facial_mask(image, landmarks, output_path="facial_mask.png"):
    """
    Creates and saves the facial mask
    """
    mask = create_facial_mask(image, landmarks)
    cv2.imwrite(output_path, mask)
    show_image("Facial Mask", mask)
    return mask


current_dir = os.getcwd()
target_path = os.path.join(current_dir, "fullbody_model.png")
# "0d384dbbcc121ca5049c423f81c26e6a.png"
source_path = os.path.join(current_dir,"0d384dbbcc121ca5049c423f81c26e6a.png" )

if not os.path.exists(source_path):
    raise FileNotFoundError(f"Source image not found: {source_path}")
if not os.path.exists(target_path):
    raise FileNotFoundError(f"Target image not found: {target_path}")

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


fullbody_img = cv2.imread("fullbody_model.png")
fullbody_face_landmarks = get_landmarks(fullbody_img)

extracted_target_face = cv2.imread('extracted_target_face.png')
extracted_target_face_landmarks = get_landmarks(extracted_target_face)
mask = save_facial_mask(extracted_target_face, extracted_target_face_landmarks)

def replace_face_in_fullbody(fullbody_image, makeup_face, face_coords):
    x1, y1, x2, y2 = face_coords

    makeup_landmarks = get_landmarks(makeup_face)
    triangles = create_triangle_mesh(makeup_landmarks, makeup_face.shape)
    croped_makeup_face = extract_face_region(makeup_face,makeup_landmarks,triangles)
    cv2.imwrite('croped_makeup_face.png',croped_makeup_face)
    # Resize the makeup face to match the original face dimensions
    face_height = y2 - y1
    face_width = x2 - x1
    makeup_face_resized = cv2.resize(croped_makeup_face, (face_width, face_height))
    
    # Create a copy of the fullbody image
    result_image = fullbody_image.copy()
    
    # Optional: Create a mask for smooth blending
    mask = np.zeros((face_height, face_width), dtype=np.float32)
    center = (face_width // 2, face_height // 2)
    radius = min(face_width, face_height) // 2
    cv2.circle(mask, center, radius, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (19, 19), 0)
    
    # Blend the face back into the image
    for c in range(3):  # for each color channel
        result_image[y1:y2, x1:x2, c] = (
            makeup_face_resized[:, :, c] * mask + 
            fullbody_image[y1:y2, x1:x2, c] * (1 - mask)
        )
    show_image('replace face in full body image',result_image)
    return result_image

# Usage example:
# fullbody_img = original full body image
# face_coords = coordinates from detect_face_mtcnn()
# makeup_face = the processed face with makeup applied

# Detect face and get coordinates
face_coords = detect_face_mtcnn(fullbody_img)
makeup_face = warped_result

# # After applying makeup to the cropped face...
result = replace_face_in_fullbody(fullbody_img, makeup_face, face_coords)



