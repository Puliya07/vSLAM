import numpy as np
import cv2
import pyrealsense2 as rs
import os
import glob
import time
import json

class RealsenseCalibrator:
    def __init__(self, output_dir='calibration_data'):
        """
        Initialize the RealSense calibrator
        
        Args:
            output_dir: Directory to save calibration data and results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Chessboard parameters
        self.chessboard_size = (9, 6)  # Number of inner corners
        self.square_size = 0.025  # Size of chessboard square in meters
        
        # Calibration parameters
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size  # Scale to actual size
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3D points in real world space
        self.rgb_imgpoints = []  # 2D points in RGB image
        self.ir_imgpoints = []  # 2D points in IR image
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
    def start_camera(self):
        """Start the RealSense camera with RGB and IR streams"""
        # Configure streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        
        # Get depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth Scale: {self.depth_scale}")
        
        # Allow autoexposure to settle
        for _ in range(5):
            self.pipeline.wait_for_frames()
            
        return self.pipeline
    
    def capture_calibration_frames(self, num_frames=20):
        """
        Capture frames for calibration
        
        Args:
            num_frames: Number of calibration frames to capture
        """
        print(f"Will capture {num_frames} frames. Press SPACE when chessboard is visible and stable.")
        
        rgb_frames = []
        ir_frames = []
        frame_count = 0
        
        while frame_count < num_frames:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(1)  # Get IR frame from first IR camera
            
            if not color_frame or not ir_frame:
                continue
                
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            ir_image = np.asanyarray(ir_frame.get_data())
            
            # Convert IR image to BGR for display
            ir_image_bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
            
            # Display images
            display_img = np.hstack((color_image, ir_image_bgr))
            cv2.putText(display_img, f"Captured: {frame_count}/{num_frames}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('RealSense Calibration (RGB + IR)', display_img)
            
            key = cv2.waitKey(1)
            
            # Capture frame when space is pressed
            if key == 32:  # Space key
                # Check if chessboard can be found in both images
                rgb_ret, _ = cv2.findChessboardCorners(color_image, self.chessboard_size, None)
                ir_ret, _ = cv2.findChessboardCorners(ir_image, self.chessboard_size, None)
                
                if rgb_ret and ir_ret:
                    # Save the frames
                    rgb_frames.append(color_image.copy())
                    ir_frames.append(ir_image.copy())
                    frame_count += 1
                    print(f"Frame {frame_count}/{num_frames} captured")
                    
                    # Save images
                    cv2.imwrite(f"{self.output_dir}/rgb_{frame_count}.png", color_image)
                    cv2.imwrite(f"{self.output_dir}/ir_{frame_count}.png", ir_image)
                    time.sleep(0.5)  # Small delay to avoid duplicates
                else:
                    print("Chessboard not detected in both images. Try again.")
            
            # Exit on ESC
            elif key == 27:
                break
        
        cv2.destroyAllWindows()
        return rgb_frames, ir_frames
    
    def find_corners(self, images, is_ir=False):
        """
        Find chessboard corners in images
        
        Args:
            images: List of images to find corners in
            is_ir: Whether the images are IR (requires different preprocessing)
            
        Returns:
            objpoints: List of object points
            imgpoints: List of image points
        """
        objpoints = []
        imgpoints = []
        
        for img in images:
            if is_ir:
                # For IR images, normalize to improve contrast
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(img, self.chessboard_size, None)
            
            if ret:
                objpoints.append(self.objp)
                
                # Refine corner positions
                gray = img if is_ir else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(corners2)
                
                # Draw corners for visualization
                vis_img = img.copy()
                if not is_ir:
                    vis_img = cv2.drawChessboardCorners(vis_img, self.chessboard_size, corners2, ret)
                    cv2.imshow('Corners', vis_img)
                    cv2.waitKey(500)
        
        cv2.destroyAllWindows()
        return objpoints, imgpoints
    
    def calibrate_camera(self, images, is_ir=False):
        """
        Calibrate camera using images
        
        Args:
            images: List of images to use for calibration
            is_ir: Whether the images are IR
            
        Returns:
            ret: RMS reprojection error
            mtx: Camera matrix
            dist: Distortion coefficients
            rvecs: Rotation vectors
            tvecs: Translation vectors
        """
        # Find corners
        objpoints, imgpoints = self.find_corners(images, is_ir)
        
        if not objpoints:
            print("No valid corners found. Calibration failed.")
            return None, None, None, None, None
        
        if is_ir:
            img_shape = images[0].shape[::-1]
        else:
            img_shape = images[0].shape[1::-1]  # Only need width and height
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None
        )
        
        print(f"{'IR' if is_ir else 'RGB'} Camera Calibration RMS Error: {ret}")
        return ret, mtx, dist, rvecs, tvecs
    
    def calibrate_stereo(self):
        """
        Calibrate stereo camera system (RGB and IR)
        
        Returns:
            rotation_matrix: Rotation matrix between RGB and IR
            translation_vector: Translation vector between RGB and IR
        """
        # Load saved calibration images
        rgb_images = []
        ir_images = []
        
        rgb_files = sorted(glob.glob(f"{self.output_dir}/rgb_*.png"))
        ir_files = sorted(glob.glob(f"{self.output_dir}/ir_*.png"))
        
        for rgb_file, ir_file in zip(rgb_files, ir_files):
            rgb_img = cv2.imread(rgb_file)
            ir_img = cv2.imread(ir_file, cv2.IMREAD_GRAYSCALE)
            
            if rgb_img is not None and ir_img is not None:
                rgb_images.append(rgb_img)
                ir_images.append(ir_img)
        
        # Calibrate individual cameras
        print("Calibrating RGB camera...")
        rgb_ret, rgb_mtx, rgb_dist, rgb_rvecs, rgb_tvecs = self.calibrate_camera(rgb_images)
        
        print("Calibrating IR camera...")
        ir_ret, ir_mtx, ir_dist, ir_rvecs, ir_tvecs = self.calibrate_camera(ir_images, is_ir=True)
        
        if rgb_ret is None or ir_ret is None:
            print("Individual camera calibration failed.")
            return None, None, None, None
        
        # Find common corners for stereo calibration
        objpoints = []
        rgb_points = []
        ir_points = []
        
        for rgb_img, ir_img in zip(rgb_images, ir_images):
            # Find RGB corners
            rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            rgb_ret, rgb_corners = cv2.findChessboardCorners(rgb_gray, self.chessboard_size, None)
            
            # Find IR corners
            ir_ret, ir_corners = cv2.findChessboardCorners(ir_img, self.chessboard_size, None)
            
            if rgb_ret and ir_ret:
                # Refine corners
                rgb_corners2 = cv2.cornerSubPix(rgb_gray, rgb_corners, (11, 11), (-1, -1), self.criteria)
                ir_corners2 = cv2.cornerSubPix(ir_img, ir_corners, (11, 11), (-1, -1), self.criteria)
                
                objpoints.append(self.objp)
                rgb_points.append(rgb_corners2)
                ir_points.append(ir_corners2)
        
        if not objpoints:
            print("No common corners found for stereo calibration.")
            return None, None, None, None
        
        # Stereo calibration
        print("Performing stereo calibration...")
        flags = 0
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints, rgb_points, ir_points,
            rgb_mtx, rgb_dist, ir_mtx, ir_dist,
            rgb_images[0].shape[1::-1],
            criteria=criteria_stereo,
            flags=flags
        )
        
        print(f"Stereo Calibration RMS Error: {ret}")
        
        # Save calibration results
        self.save_calibration_results(rgb_mtx, rgb_dist, ir_mtx, ir_dist, R, T)
        
        return rgb_mtx, rgb_dist, ir_mtx, ir_dist, R, T
    
    def save_calibration_results(self, rgb_mtx, rgb_dist, ir_mtx, ir_dist, R, T):
        """
        Save calibration results to file
        
        Args:
            rgb_mtx: RGB camera matrix
            rgb_dist: RGB distortion coefficients
            ir_mtx: IR camera matrix
            ir_dist: IR distortion coefficients
            R: Rotation matrix
            T: Translation vector
        """
        calibration_data = {
            "rgb_camera_matrix": rgb_mtx.tolist(),
            "rgb_distortion_coefficients": rgb_dist.tolist(),
            "ir_camera_matrix": ir_mtx.tolist(),
            "ir_distortion_coefficients": ir_dist.tolist(),
            "rotation_matrix": R.tolist(),
            "translation_vector": T.tolist(),
            "depth_scale": self.depth_scale
        }
        
        with open(f"{self.output_dir}/calibration_results.json", 'w') as f:
            json.dump(calibration_data, f, indent=4)
        
        print(f"Calibration results saved to {self.output_dir}/calibration_results.json")
    
    def load_calibration(self, file_path=None):
        """
        Load calibration data from file
        
        Args:
            file_path: Path to calibration file, defaults to output_dir/calibration_results.json
            
        Returns:
            dict: Calibration data
        """
        if file_path is None:
            file_path = f"{self.output_dir}/calibration_results.json"
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def undistort_images(self, rgb_img, ir_img, calib_data=None):
        """
        Undistort images using calibration data
        
        Args:
            rgb_img: RGB image
            ir_img: IR image
            calib_data: Calibration data, loaded from file if None
            
        Returns:
            tuple: Undistorted RGB and IR images
        """
        if calib_data is None:
            calib_data = self.load_calibration()
        
        # Convert lists back to numpy arrays
        rgb_mtx = np.array(calib_data["rgb_camera_matrix"])
        rgb_dist = np.array(calib_data["rgb_distortion_coefficients"])
        ir_mtx = np.array(calib_data["ir_camera_matrix"])
        ir_dist = np.array(calib_data["ir_distortion_coefficients"])
        
        # Undistort images
        rgb_undistorted = cv2.undistort(rgb_img, rgb_mtx, rgb_dist)
        ir_undistorted = cv2.undistort(ir_img, ir_mtx, ir_dist)
        
        return rgb_undistorted, ir_undistorted
    
    def run_calibration(self, num_frames=20):
        """
        Run the complete calibration process
        
        Args:
            num_frames: Number of frames to capture for calibration
        """
        print("Starting RealSense D435i calibration...")
        
        # Start camera
        self.start_camera()
        
        # Capture calibration frames
        print("Capturing calibration frames...")
        rgb_frames, ir_frames = self.capture_calibration_frames(num_frames)
        
        # Perform stereo calibration
        print("Performing calibration...")
        calibration_results = self.calibrate_stereo()
        
        # Stop pipeline
        self.pipeline.stop()
        
        if calibration_results[0] is not None:
            print("Calibration completed successfully!")
        else:
            print("Calibration failed.")
    
    def verify_calibration(self):
        """
        Verify calibration by displaying undistorted images
        """
        # Start camera
        self.start_camera()
        
        try:
            # Load calibration data
            calib_data = self.load_calibration()
            
            print("Press ESC to exit calibration verification")
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                ir_frame = frames.get_infrared_frame(1)
                
                if not color_frame or not ir_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                ir_image = np.asanyarray(ir_frame.get_data())
                
                # Undistort images
                rgb_undistorted, ir_undistorted = self.undistort_images(color_image, ir_image, calib_data)
                
                # Convert IR image to BGR for display
                ir_image_bgr = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
                ir_undistorted_bgr = cv2.cvtColor(ir_undistorted, cv2.COLOR_GRAY2BGR)
                
                # Create display images
                original = np.hstack((color_image, ir_image_bgr))
                undistorted = np.hstack((rgb_undistorted, ir_undistorted_bgr))
                
                # Add labels
                cv2.putText(original, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(undistorted, "Undistorted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display images
                cv2.imshow('Calibration Verification', np.vstack((original, undistorted)))
                
                if cv2.waitKey(1) == 27:  # ESC
                    break
                
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrator = RealsenseCalibrator()
    
    # Run the calibration process
    calibrator.run_calibration(num_frames=20)
    
    # Verify calibration
    input("Press Enter to verify calibration...")
    calibrator.verify_calibration()