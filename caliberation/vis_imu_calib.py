import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

# Function to detect checkerboard in images
def detect_checkerboard(image_folder, pattern_size=(9, 6)):
    """
    Detect checkerboard corners in all images in the folder.
    Returns detected corners and corresponding image paths.
    """
    image_paths = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    detected_corners = []
    successful_images = []
    
    # Criteria for corner detection
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    # Scale to actual size (assume checker size is 20mm)
    square_size = 0.02  # 20mm
    objp *= square_size
    
    obj_points = []
    img_points = []
    
    for image_path in image_paths:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            # Refine corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Store result
            obj_points.append(objp)
            img_points.append(corners2)
            successful_images.append(image_path)
            
            # Draw corners on image and save
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            detected_filename = os.path.join('calibration_data/detected', 
                                            os.path.basename(image_path))
            if not os.path.exists('calibration_data/detected'):
                os.makedirs('calibration_data/detected')
            cv2.imwrite(detected_filename, img)
    
    print(f"Successfully detected corners in {len(successful_images)}/{len(image_paths)} images")
    return obj_points, img_points, successful_images

# Camera calibration function
def calibrate_camera(obj_points, img_points, img_shape):
    """
    Calibrate camera using detected checkerboard corners.
    Returns camera matrix, distortion coefficients, rotation and translation vectors.
    """
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], 
                                          camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        mean_error += error
    
    mean_error /= len(obj_points)
    print(f"Camera calibration reprojection error: {mean_error}")
    
    return camera_matrix, dist_coeffs, rvecs, tvecs

# Function to synchronize camera and IMU data
def synchronize_camera_imu(imu_data, successful_images):
    """
    Synchronize IMU data with camera frames.
    Returns matched IMU measurements for each camera frame.
    """
    image_timestamps = [float(os.path.basename(path).split('.')[0]) / 1000 
                      for path in successful_images]
    
    # IMU data format: timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    imu_timestamps = imu_data[:, 0]
    
    # Find closest IMU measurements for each image
    matched_imu = []
    for img_time in image_timestamps:
        idx = np.argmin(np.abs(imu_timestamps - img_time))
        matched_imu.append(imu_data[idx, 1:])
    
    return np.array(matched_imu)

# Visual-inertial calibration
def visual_inertial_calibration(camera_matrix, dist_coeffs, rvecs, tvecs, matched_imu):
    """
    Calibrate the transformation between camera and IMU.
    Returns rotation and translation between camera and IMU.
    """
    # Define objective function for optimization
    def objective_function(params):
        R_ci = Rotation.from_rotvec(params[:3]).as_matrix()
        t_ci = params[3:6].reshape(3, 1)
        
        residuals = []
        for i in range(len(rvecs)):
            # Camera pose in world frame
            R_wc = Rotation.from_rotvec(rvecs[i]).as_matrix()
            t_wc = tvecs[i].reshape(3, 1)
            
            # IMU measurements (acceleration and angular velocity)
            accel = matched_imu[i, :3].reshape(3, 1)
            gyro = matched_imu[i, 3:].reshape(3, 1)
            
            # Transform IMU measurements to camera frame
            accel_cam = R_ci @ accel + t_ci
            gyro_cam = R_ci @ gyro
            
            # Expected measurements based on camera pose change
            if i > 0:
                dt = 1/30.0  # Assuming 30 fps
                R_prev = Rotation.from_rotvec(rvecs[i-1]).as_matrix()
                t_prev = tvecs[i-1].reshape(3, 1)
                
                # Angular velocity from rotation change
                dR = R_wc @ R_prev.T
                omega = Rotation.from_matrix(dR).as_rotvec() / dt
                
                # Linear acceleration from position change
                dv = (t_wc - t_prev) / dt
                a = dv / dt
                
                # Calculate residuals
                residuals.extend((gyro_cam - omega.reshape(3, 1)).flatten())
                residuals.extend((accel_cam - a - np.array([[0], [0], [9.81]])).flatten())
        
        return np.array(residuals)
    
    # Initial guess: identity rotation and zero translation
    initial_params = np.zeros(6)
    
    # Run optimization
    result = least_squares(objective_function, initial_params, method='lm', verbose=2)
    
    # Extract results
    R_ci = Rotation.from_rotvec(result.x[:3])
    t_ci = result.x[3:6]
    
    return R_ci, t_ci

def main():
    # Create output directory for detected checkerboards
    if not os.path.exists('calibration_data/detected'):
        os.makedirs('calibration_data/detected')
    
    # 1. Detect checkerboard in images
    print("Detecting checkerboard patterns...")
    obj_points, img_points, successful_images = detect_checkerboard('calibration_data/images')
    
    if len(successful_images) < 10:
        print("Not enough images with detected checkerboard. Need at least 10.")
        return
    
    # 2. Camera calibration
    print("Calibrating camera...")
    img = cv2.imread(successful_images[0])
    img_shape = (img.shape[1], img.shape[0])
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(obj_points, img_points, img_shape)
    
    # 3. Load IMU data
    print("Loading IMU data...")
    imu_data = np.loadtxt('calibration_data/imu_data.csv', delimiter=',', skiprows=1)
    
    # 4. Synchronize camera and IMU data
    print("Synchronizing camera and IMU data...")
    matched_imu = synchronize_camera_imu(imu_data, successful_images)
    
    # 5. Visual-inertial calibration
    print("Performing visual-inertial calibration...")
    R_ci, t_ci = visual_inertial_calibration(camera_matrix, dist_coeffs, rvecs, tvecs, matched_imu)
    
    # 6. Save calibration results
    print("Saving calibration results...")
    calibration_results = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
        'R_camera_imu': R_ci.as_matrix().tolist(),
        't_camera_imu': t_ci.tolist()
    }
    
    np.save('calibration_data/calibration_results.npy', calibration_results)
    
    # 7. Print results
    print("\nCalibration Results:")
    print("Camera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)
    print("\nCamera-IMU Rotation (as Euler angles in degrees):")
    print(R_ci.as_euler('xyz', degrees=True))
    print("\nCamera-IMU Translation (in meters):")
    print(t_ci)
    
    print("\nCalibration complete! Results saved to calibration_data/calibration_results.npy")

if __name__ == "__main__":
    main()