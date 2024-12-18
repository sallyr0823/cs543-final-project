import numpy as np
import cv2
import face_detection_pipeline as FacePipeline

class FaceBlender:
    @staticmethod
    def calculate_triangle_area(v1,v2,v3):
        x1, y1 = v1
        x2, y2 = v2
        x3, y3 = v3
        return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))/2.0)

    @staticmethod
    def compute_barycentric_coordinates(point, triangle_vertices):
        """Compute barycentric coordinates of a point relative to a triangle"""
        v1, v2, v3 = triangle_vertices
        total_area = FaceBlender.calculate_triangle_area(v1, v2, v3)
        if total_area == 0:
            return np.array([0, 0, 0])
            
        w1 = FaceBlender.calculate_triangle_area(point, v2, v3) / total_area
        w2 = FaceBlender.calculate_triangle_area(point, v3, v1) / total_area
        w3 = FaceBlender.calculate_triangle_area(point, v1, v2) / total_area
        
        return np.array([w1, w2, w3])

    def alpha_blend_images(self, source_image, target_image,alpha_1, alpha_2, source_tuple, target_tuple, skin_only=False):
        """
        Morph between source and target images using feature points and triangulation
        
        Args:
            source_image: Original image to be blended
            target_image: Target image with features to be taken
            alpha_1: Weight for source image
            alpha_2: Weight for target image
            source_tuple: (output_image, feature_points, triangulation, mask) for source
            target_tuple: (output_image, feature_points, triangulation, mask) for target
            skin_only: If True, only morph skin regions
        """
        # Unpack tuples
        _, source_points, source_triangulation,source_mask = source_tuple
        _, target_points, _,target_mask = target_tuple
        
        # Initialize output image
        blended_image = np.copy(source_image)
        
        # Process each triangle in the triangulation
        for idx, triangle in enumerate(source_triangulation.simplices):
            # Get triangle vertices in both images
            source_vertices = [source_points[i] for i in triangle]
            target_vertices = [target_points[i] for i in triangle]
            
            # Get bounding box for the triangle
            s_x = [v[0] for v in source_vertices]
            s_y = [v[1] for v in source_vertices]
            min_x, max_x = min(s_x), max(s_x)
            min_y, max_y = min(s_y), max(s_y)
            
            # Process each pixel in the bounding box
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    # Skip if not in skin region when skin_only is True
                    if skin_only and not (source_mask[y][x] in [1, 2]):
                        continue
                        
                    point = np.array([x, y])
                    
                    # Check if point is in current triangle
                    if source_triangulation.find_simplex(point) == idx:
                        # Calculate barycentric coordinates
                        weights = self.compute_barycentric_coordinates(
                            point, source_vertices
                        )
                        
                        # Find corresponding point in target image
                        target_point = np.sum([w * v for w, v in zip(weights, target_vertices)], axis=0)
                        target_point = target_point.astype(int)
                        
                        if skin_only and source_mask[y][x] == 2:  # Special handling for lips
                            blended_image[y, x] = target_image[target_point[1], target_point[0]]
                        else:
                            blended_image[y, x] = (
                                alpha_2 * target_image[target_point[1], target_point[0]] +
                                alpha_1 * source_image[y, x]
                            )
        
        return blended_image

def main():
    # Initialize morpher
    blender = FaceBlender()
    
    # Load and resize source image
    source_path = "test.png"
    source_image = cv2.imread(source_path)
    if source_image is None:
        raise FileNotFoundError("Source image not found")
    
    # Load and resize target image
    target_path = "target.png"
    target_image = cv2.imread(target_path)
    if target_image is None:
        raise FileNotFoundError("Target image not found")

    
    # Detect landmarks and get feature points
    l = FacePipeline.FaceLandmarkDetector()
    source_tuple =l.landmark_detection(source_image)
    target_tuple = l.landmark_detection(target_image)
    
    alpha1 = 0.2  # Weight for source image
    alpha2 = 0.8  # Weight for target image
    blended_image = blender.blend_images(
        source_image, 
        target_image,
        alpha1,
        alpha2,
        source_tuple,
        target_tuple
    )
    
    # Display results
    result = np.hstack((source_image, blended_image,target_image))
    cv2.imshow("Face Blending Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()