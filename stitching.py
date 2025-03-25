import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob

def display_image(img, title=""):
    plt.figure()
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
        plt.colorbar()
        plt.title(title)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(title)
    else:
        return
    plt.show()


def load_images(dir="../data/Image_Registration"):
    images = sorted(glob(os.path.join(dir, "*.png")))
    images = [cv2.imread(i) for i in images]
    return images

def to_grayscale(func):
    def func_grayscale(image, *args, **kwargs):
        if image.ndim == 2:
            return func(image, *args,  **kwargs)
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return func(gray, *args,  **kwargs)
        else:
            return func(image, *args,  **kwargs)
    return func_grayscale

@to_grayscale
def extract_features(image):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)

    # orb = cv2.ORB_create(5000)
    # kp, des = orb.detectAndCompute(image, None)
    return kp, des


def match_features(reference, image):
    QUALITY_RATIO = 0.7
    MIN_MATCH_COUNT = 10

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    kp_ref, des_ref = extract_features(reference)
    kp_image, des_image = extract_features(image)

    matches = flann.knnMatch(des_ref, des_image, k=2)
    # matches = bf.knnMatch(des_ref, des_image, k=2)

    good = []
    for m, n in matches:
        if m.distance < QUALITY_RATIO * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        image_pts = np.float32([kp_image[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return ref_pts, image_pts
    else:
        raise RuntimeError(f"Not enough good matches to proceed with homography estimation! "
                           f"Min: {MIN_MATCH_COUNT}, got {len(good)}.")


def estimate_homography(reference, image):
    ref_pts, image_pts = match_features(reference, image)
    homography, _ = cv2.findHomography(image_pts, ref_pts, cv2.RANSAC, 5.0)
    return homography

def warp_image(image, homography, output_size):
    warped = cv2.warpPerspective(image, homography, output_size)
    return warped

@to_grayscale
def warp_corners(image, homography):
    h, w = image.shape
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T  # 3x4 matrix
    transformed_corners = homography @ corners  # Apply homography
    transformed_corners /= transformed_corners[2]  # Normalize by z
    return transformed_corners[:2].T  # Return (x, y) coordinates

def compute_offset(images, homographies):
    """Find the translation needed to keep all warped images in positive coordinates."""
    all_corners = np.vstack([warp_corners(image, homography) for image, homography in zip(images, homographies)])
    x_min, y_min = np.min(all_corners, axis=0).astype(int)  # Find min x, y
    return -x_min, -y_min  # Compute translation needed to shift images to (0,0)

def adjust_homographies(images, homographies):
    dx, dy = compute_offset(images, homographies)
    T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])  # Translation matrix
    return [T @ H for H in homographies]  # Apply translation

def compute_output_size(images, homographies):
    """Find the bounding box for all transformed images."""
    all_corners = np.vstack([warp_corners(image, homography) for image, homography in zip(images, homographies)])

    x_min, y_min = np.min(all_corners, axis=0).astype(int)
    x_max, y_max = np.max(all_corners, axis=0).astype(int)

    return x_max - x_min, y_max - y_min

def _swap(x: tuple) -> tuple:
    if len(x) == 2:
        return x[1], x[0]

def create_canvas(output_size):
    return np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)

def stitch(reference, images, homographies):
    output_size = compute_output_size(images, homographies)

    images = [reference] + images
    ref_homography = np.eye(3, dtype=np.float32)
    homographies = [ref_homography] + homographies

    homographies = adjust_homographies(images, homographies)

    canvas = np.zeros((*_swap(output_size), 3), dtype=np.float32)
    mask = np.zeros((*_swap(output_size), 3), dtype=np.float32)

    for image, homography in zip(images, homographies):
        warped = warp_image(image, homography, output_size)
        warped_mask = (warped > 0).astype(np.float32)  # Mask of valid pixels

        # Add warped image contribution
        canvas += warped * warped_mask
        mask += warped_mask  # Track how many images contribute per pixel

    # Normalize overlap areas to avoid intensity buildup
    mask[mask == 0] = 1  # Prevent division by zero
    canvas /= mask

    return np.uint8(canvas)  # Convert back to uint8 for display


def draw_matches(reference, image):
    kp_ref, des_ref = extract_features(reference)
    kp_image, des_image = extract_features(image)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_ref, des_image, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]  # Same ratio used in matching

    img_matches = cv2.drawMatches(reference, kp_ref, image, kp_image, good, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    display_image(img_matches, title="Feature Matches")


def debug():
    images = load_images()
    reference = images[0]
    images = images[1:]

    for i, image in enumerate(images):
        draw_matches(reference, image)  # Debug step

        homography = estimate_homography(reference, image)

        print(homography)

        warped = warp_image(image, homography, (reference.shape[1], reference.shape[0]))
        display_image(warped, title=f"Warped Image {i+1}")


def validate():
    images = load_images()
    reference = images[0]
    images = images[1:]

    homographies = []
    for i, image in enumerate(images):
        homography = estimate_homography(reference, image)
        homographies.append(homography)
        # Warp the image using estimated homography
        warped = warp_image(image, homography, (reference.shape[1], reference.shape[0]))
        # Overlay the warped image on the reference
        overlay = cv2.addWeighted(reference, 0.5, warped, 0.5, 0)
        # Show the overlayed result
        display_image(overlay, title=f"Warped Image {i+1} Over Reference")


def main():
    images = load_images()
    reference = images[0]
    images = images[1:]

    homographies = []
    for image in images:
        homography = estimate_homography(reference, image)
        homographies.append(homography)

    canvas = stitch(reference, images, homographies)

    display_image(canvas)


if __name__ == "__main__":
    # debug()
    # validate()
    main()