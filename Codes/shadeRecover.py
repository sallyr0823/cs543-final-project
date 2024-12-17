import cv2
import matplotlib.pyplot as plt
import numpy as np



def shadeRecover(sourceImage,resultant_image,makeUpImage):

    # Convert images to RGB for display
    sourceImage_rgb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2RGB)
    resultant_image_rgb = cv2.cvtColor(resultant_image, cv2.COLOR_BGR2RGB)
    makeUpImage_rgb = cv2.cvtColor(makeUpImage, cv2.COLOR_BGR2RGB)

        # Apply color transfer from original (left) to result (center)
    corrected_center_image = color_transfer(sourceImage_rgb, resultant_image_rgb)

    # Convert to RGB for visualization
    corrected_center_image_rgb = cv2.cvtColor(corrected_center_image, cv2.COLOR_BGR2RGB)

    # Display the corrected center image
    cv2.imshow("corrected shadow", corrected_center_image_rgb)

    # Save the corrected image for user to download
    output_path = 'corrected_result.png'
    cv2.imwrite(output_path, corrected_center_image)


def color_transfer(source, target):
    """
    Transfers color from source to target using mean and standard deviation in LAB color space.
    """
    # Convert both images to LAB color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # Compute mean and std for source and target in LAB space
    (l_mean_src, a_mean_src, b_mean_src), (l_std_src, a_std_src, b_std_src) = cv2.meanStdDev(source_lab)
    (l_mean_tar, a_mean_tar, b_mean_tar), (l_std_tar, a_std_tar, b_std_tar) = cv2.meanStdDev(target_lab)

    # Subtract target mean and scale by std deviation
    (l, a, b) = cv2.split(target_lab)
    l -= l_mean_tar
    a -= a_mean_tar
    b -= b_mean_tar

    l = (l * (l_std_src / l_std_tar)) + l_mean_src
    a = (a * (a_std_src / a_std_tar)) + a_mean_src
    b = (b * (b_std_src / b_std_tar)) + b_mean_src

    # Clip and merge back
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    result_lab = cv2.merge([l, a, b])

    # Convert back to BGR
    result = cv2.cvtColor(result_lab.astype("uint8"), cv2.COLOR_LAB2BGR)
    return result

