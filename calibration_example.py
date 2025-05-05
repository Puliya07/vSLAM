import cv2
import numpy as np
import pyrealsense2 as rs
import json
import os
import time

def load_calibration(file_path):
    """
    Load calibration data from file
    
    Args:
        file_path: Path to calibration file
        
    Returns:
        dict: Calibration data
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data

def apply_calibration_to_pointcloud(file_path='calibration_data/calibration_results.json'):
    """
    Apply calibration to depth data and generate a corrected point cloud
    
    Args:
        file_path: Path to calibration file
    """
    try:
        # Load calibration data
        calib_data = load_calibration(file_path)
        
        # Convert lists back to numpy arrays
        rgb_mtx = np.array(calib_data["rgb_camera_matrix"])
        rgb_dist = np.array(calib_data["rgb_distortion_coefficients"])
        ir_mtx = np.array(calib_data["ir_camera_matrix"])
        ir_dist = np.array(calib_data["ir_distortion_coefficients"])
        rotation_matrix = np.array(calib_data["rotation_matrix"])
        translation_vector = np.array(calib_data["translation_vector"])
        depth_scale = calib_data["depth_scale"]
        
        # Initialize RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable streams
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start streaming
        profile = pipeline.start(config)
        
        # Create align object
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        # Get depth intrinsics
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        
        # Use loaded calibration data to create custom intrinsics
        depth_intrinsics.width = 640
        depth_intrinsics.height = 480
        depth_intrinsics.ppx = ir_mtx[0, 2]
        depth_intrinsics.ppy = ir_mtx[1, 2]
        depth_intrinsics.fx = ir_mtx[0, 0]
        depth_intrinsics.fy = ir_mtx[1, 1]
        
        # Create a point cloud object
        pc = rs.pointcloud()
        
        try:
            print("Press ESC to exit point cloud visualization")
            
            while True:
                # Wait for frames
                frames = pipeline.wait_for_frames()
                
                # Align frames
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Apply undistortion to color image
                undistorted_color = cv2.undistort(color_image, rgb_mtx, rgb_dist)
                
                # Map texture onto point cloud
                pc.map_to(color_frame)
                
                # Generate point cloud
                points = pc.calculate(depth_frame)
                
                # Display color image with depth overlay
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # Create a blended image
                alpha = 0.6
                blended_image = cv2.addWeighted(
                    undistorted_color, alpha, 
                    depth_colormap, 1 - alpha, 
                    0
                )
                
                # Display image
                cv2.putText(
                    blended_image, 
                    "Calibrated Point Cloud Visualization", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 255, 255), 
                    2
                )
                
                cv2.imshow('Calibrated RealSense', blended_image)
                
                if cv2.waitKey(1) == 27:  # ESC
                    break
                    
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            
    except FileNotFoundError:
        print(f"Calibration file not found: {file_path}")
        print("Please run calibration first.")
    except Exception as e:
        print(f"Error applying calibration: {e}")

if __name__ == "__main__":
    if os.path.exists('calibration_data/calibration_results.json'):
        print("Using existing calibration data...")
        apply_calibration_to_pointcloud()
    else:
        print("No calibration data found. Please run calibration first.")
        print("Execute: python realsense_d435i_calibration.py")