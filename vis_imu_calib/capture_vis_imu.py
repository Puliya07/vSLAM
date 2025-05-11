#!/usr/bin/env python3
import os, csv, time
import cv2
import numpy as np
import pyrealsense2 as rs

# === USER PARAMETERS ===
OUTPUT_DIR   = "calib_dataset"
IMG_DIR      = os.path.join(OUTPUT_DIR, "cam0")
IMU_DIR      = os.path.join(OUTPUT_DIR, "imu0")
FPS          = 30               # camera FPS
DURATION     = 60               # record for 60 seconds
CHECKER_SIZE = (9, 6)           # inner corners
# =========================

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(IMU_DIR, exist_ok=True)

# CSV writers
imu_f = open(os.path.join(IMU_DIR, "data.csv"), "w", newline="")
cam_f = open(os.path.join(IMG_DIR, "data.csv"), "w", newline="")
imu_w = csv.writer(imu_f); imu_w.writerow(["t","gx","gy","gz","ax","ay","az"])
cam_w = csv.writer(cam_f); cam_w.writerow(["t","filename"])

# Configure RealSense
pipe = rs.pipeline(); cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, FPS)
cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, FPS*4)
cfg.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f, FPS*4)
profile = pipe.start(cfg)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
start = time.time(); idx=0; last_gyro=None

print("Recording… show checkerboard in view!")
try:
    while time.time() - start < DURATION:
        frames = pipe.wait_for_frames()
        # ==== IMU Logging ====
        for f in frames:
            st = f.get_profile().stream_type()
            if st == rs.stream.gyro:
                last_gyro = f.as_motion_frame().get_motion_data()
            elif st == rs.stream.accel and last_gyro is not None:
                a = f.as_motion_frame().get_motion_data()
                gx,gy,gz = last_gyro; ax,ay,az = a
                imu_w.writerow([f.get_timestamp()/1000.0, gx,gy,gz, ax,ay,az])

        # ==== Checkerboard Detection ====
        color = frames.get_color_frame()
        if not color: continue
        img = np.asanyarray(color.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKER_SIZE, None)
        if found:
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(img, CHECKER_SIZE, corners, found)
            cv2.imshow("Detect", img); cv2.waitKey(1)

            fname = f"{idx:06d}.png"
            cv2.imwrite(os.path.join(IMG_DIR, fname), img)
            cam_w.writerow([color.get_timestamp()/1000.0, fname])
            idx += 1
        time.sleep(0.01)
finally:
    pipe.stop()
    imu_f.close(); cam_f.close()
    cv2.destroyAllWindows()
    print(f"Done: {idx} images captured.")
