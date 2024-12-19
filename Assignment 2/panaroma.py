import numpy as np
import cv2
import sys
import glob
import matplotlib.pyplot as plt

image_paths = glob.glob(r"New Folder\*.jpg")
if not image_paths:
    sys.exit("Error: No images found in the directory")

print(image_paths)

# Load images
images = []
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not load image {path}")
        continue
    images.append(img)

# Convert to grayscale, detect keypoints and descriptors
sift = cv2.SIFT_create()
keypoints = []
descriptors = []
count = 0
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kpts, dpts = sift.detectAndCompute(gray, None)
    # show the detected features overlaid on the images. Also give out the number of features detected in each image.
    img1=cv2.drawKeypoints(img,kpts, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # count += 1
    # plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    # plt.title(f"Image {count} with keypoints detected")
    # plt.axis('off')
    # plt.show()
    print(f"Number of features detected in image {len(keypoints)}: {len(kpts)}")
    keypoints.append(kpts)
    descriptors.append(dpts)

# Match features
bf = cv2.BFMatcher()
good_matches = []

for i in range(len(descriptors) - 1):
    matches = bf.knnMatch(descriptors[i], descriptors[i + 1], k=2)
    print(f"Number of matches between images {i} and {i+1}: {len(matches)}")
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:  # Ratio test
            good.append([m])
    good_matches.append(good)

    #draw top 10 matchesKNN
    #sort matches based on distance
    # good = sorted(good, key = lambda x:x[0].distance)
    # img_matches = cv2.drawMatchesKnn(images[i], keypoints[i], images[i+1], keypoints[i+1], good[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    # plt.title(f"Top 10 matches before RANSAC between images {i} and {i+1}")
    # plt.axis('off')
    # plt.show()

# Find homography
homographies = []
for i in range(len(good_matches)):
    if len(good_matches[i]) >= 4:
        src_pts = np.float32([keypoints[i][m[0].queryIdx].pt for m in good_matches[i]]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[i + 1][m[0].trainIdx].pt for m in good_matches[i]]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        #Show total number of inlier matches after homography estimations. Also show top 10 matches that have the minimum error between the projected source keypoint and the destination keypoint. 
        inlier_count = np.sum(mask)
        print(f"Number of inlier matches between images {i} and {i+1}: {inlier_count}")
        
        # Reprojection error calculation for each inlier
        def reprojection_error(src_pt, dst_pt, H):
            src_pt_proj = np.dot(H, np.array([src_pt[0], src_pt[1], 1]))
            src_pt_proj /= src_pt_proj[2]  # Normalize
            error = np.linalg.norm(src_pt_proj[:2] - dst_pt)
            return error
        
        inlier_matches = [m for j, m in enumerate(good_matches[i]) if mask[j] == 1]
        errors = []
        for m in inlier_matches:
            src_pt = keypoints[i][m[0].queryIdx].pt
            dst_pt = keypoints[i+1][m[0].trainIdx].pt
            error = reprojection_error(src_pt, dst_pt, H)
            errors.append((m, error))
        
        # Sort based on reprojection error and get the top 10 matches with least error
        errors.sort(key=lambda x: x[1])
        top_10_matches = [e[0] for e in errors[:10]]

        # Display top 10 matches with minimum reprojection error
        # img_top_matches = cv2.drawMatchesKnn(images[i], keypoints[i], images[i+1], keypoints[i+1], top_10_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # plt.imshow(cv2.cvtColor(img_top_matches, cv2.COLOR_BGR2RGB))
        # plt.title(f"Top 10 inlier matches with minimum error between images {i} and {i+1}")
        # plt.axis('off')
        # plt.show()

        print(H)
        homographies.append(H)
    else:
        print(f"Warning: Not enough matches between images {i} and {i+1}")
        homographies.append(None)

count = len(images)
idx_center = count // 2

cumulative_homographies = [np.eye(3) for _ in range(count)]
for i in range(idx_center-1, -1, -1):
    cumulative_homographies[i] = np.dot(homographies[i], cumulative_homographies[i+1])
for i in range(idx_center, count-1):
    cumulative_homographies[i+1] = np.dot(np.linalg.inv(homographies[i]), cumulative_homographies[i])


min_x = min_y = max_x = max_y = 0.0
for i in range(len(images)):
    h, w, p = images[i].shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    transformed_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), cumulative_homographies[i])
    min_x = min(transformed_corners[:, 0, 0].min(), min_x)
    min_y = min(transformed_corners[:, 0, 1].min(), min_y)
    max_x = max(transformed_corners[:, 0, 0].max(), max_x)
    max_y = max(transformed_corners[:, 0, 1].max(), max_y)

output_width = int(max_x - min_x)
output_height = int(max_y - min_y)

panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)
warped_images = []

for i in range(len(images)):
    offset_x = int(-min_x)
    offset_y = int(-min_y)
    transformation = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
    warped = cv2.warpPerspective(images[i], np.dot(transformation, cumulative_homographies[i]), (output_width, output_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped_images.append(warped)
    mask = np.where(warped > 0, 1, 0).astype(np.uint8)
    panorama_mask = np.where(panorama > 0, 1, 0).astype(np.uint8) 
    panorama = np.where(mask == 1, warped, panorama)
    alpha = 0.5
    panorama = np.where((mask == 1) & (panorama_mask == 1), (warped * alpha + panorama * (1-alpha)).astype(np.uint8), panorama)

fig = plt.figure(figsize=(10, 8), dpi=200)

plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.title("Panorama", fontsize=8)
plt.axis('off')
plt.show()

# for i in range(len(warped_images)):
#     fig = plt.figure(figsize=(10, 8), dpi=200)
#     plt.imshow(cv2.cvtColor(warped_images[i], cv2.COLOR_BGR2RGB))
#     plt.title(f"Transformed Image {i+1}", fontsize=8)
#     plt.axis('off')
#     plt.show()