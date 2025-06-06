import cv2
import numpy as np
import glob
import yaml


CHECKERBOARD = (9, 5)  
SQUARE_SIZE = 25.0  

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE


objpoints = []  # 3D points
imgpoints = []  # 2D points

# Load images
images = glob.glob('calibration_images/*.[jp][pn][gf]')
if not images:
    print("Error: No images found in calibration_images/")
    exit()

for fname in images:
    print(f"Processing {fname}...")
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners)
    else:
        print(f"Checkerboard not found in {fname}")

if len(objpoints) == 0:
    print("Error: No valid checkerboard images found.")
    exit()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("Calibration successful!")
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)


    calibration_data = {
        'camera_matrix': mtx.tolist(),
        'dist_coeff': dist.tolist()
    }
    with open('camera_params.yaml', 'w') as f:
        yaml.dump(calibration_data, f)
    print("Saved to camera_params.yaml")


    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"Mean reprojection error: {mean_error / len(objpoints):.3f} pixels")
else:
    print("Calibration failed.")