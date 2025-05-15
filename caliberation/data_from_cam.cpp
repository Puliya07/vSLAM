#include <librealsense2/rs.hpp>
#include <iostream>
#include <iomanip>

int main() {
    try {
        rs2::context ctx;
        rs2::pipeline pipe;
        rs2::config cfg;

        // Enable streams
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
        cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 250);
        cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 400);

        // Start pipeline
        rs2::pipeline_profile profile = pipe.start(cfg);

        std::cout << "Streaming RGB-D and IMU data from D435i...\n";

        while (true) {
            rs2::frameset frames = pipe.wait_for_frames();

            // RGB and Depth
            rs2::video_frame color = frames.get_color_frame();
            rs2::depth_frame depth = frames.get_depth_frame();

            if (color && depth) {
                std::cout << "Color frame size: "
                          << color.get_width() << "x" << color.get_height() << " | "
                          << "Depth frame size: "
                          << depth.get_width() << "x" << depth.get_height() << "\n";
            }

            // IMU: gyro and accel come separately
            for (auto &&frame : frames) {
                if (auto motion = frame.as<rs2::motion_frame>()) {
                    rs2_vector data = motion.get_motion_data();
                    if (motion.get_profile().stream_type() == RS2_STREAM_GYRO) {
                        std::cout << std::fixed << std::setprecision(3)
                                  << "Gyro: [" << data.x << ", " << data.y << ", " << data.z << "]\n";
                    } else if (motion.get_profile().stream_type() == RS2_STREAM_ACCEL) {
                        std::cout << std::fixed << std::setprecision(3)
                                  << "Accel: [" << data.x << ", " << data.y << ", " << data.z << "]\n";
                    }
                }
            }
        }
    } catch (const rs2::error &e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << "Other error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
