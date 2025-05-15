#include <librealsense2/rs.hpp>
#include <iostream>
#include <iomanip>

int main() {
    try {
        // First, check if any devices are connected
        rs2::context ctx;
        auto devices = ctx.query_devices();
        
        if (devices.size() == 0) {
            std::cerr << "Error: No RealSense devices detected. Please connect a camera and try again." << std::endl;
            return EXIT_FAILURE;
        }
        
        // Print device information
        std::cout << "Found " << devices.size() << " RealSense device(s):" << std::endl;
        for (size_t i = 0; i < devices.size(); i++) {
            auto device = devices[i];
            std::cout << "  " << i+1 << ": " << device.get_info(RS2_CAMERA_INFO_NAME) 
                    << " (S/N: " << device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << ")" << std::endl;
        }
        
        // Create pipeline
        rs2::pipeline pipe;
        rs2::config cfg;
        
        // Start with just basic streams that all cameras support
        std::cout << "Configuring basic color and depth streams..." << std::endl;
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
        
        // Try to start the pipeline with this basic configuration
        std::cout << "Starting pipeline..." << std::endl;
        rs2::pipeline_profile profile;
        
        try {
            profile = pipe.start(cfg);
            std::cout << "Pipeline started successfully!" << std::endl;
        } catch (const rs2::error& e) {
            std::cerr << "Failed to start pipeline with color and depth: " << e.what() << std::endl;
            std::cout << "Trying with only depth stream..." << std::endl;
            
            // Try with just depth
            cfg = rs2::config(); // Reset config
            cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
            
            try {
                profile = pipe.start(cfg);
                std::cout << "Pipeline started with depth only!" << std::endl;
            } catch (const rs2::error& e) {
                std::cerr << "Failed to start pipeline with any configuration: " << e.what() << std::endl;
                std::cout << "Trying with default settings..." << std::endl;
                
                try {
                    // Try with no config at all (uses default)
                    profile = pipe.start();
                    std::cout << "Pipeline started with default settings!" << std::endl;
                } catch (const rs2::error& e) {
                    std::cerr << "Failed to start pipeline with default settings: " << e.what() << std::endl;
                    return EXIT_FAILURE;
                }
            }
        }
        
        // Get active device
        auto active_device = profile.get_device();
        std::cout << "Using device: " << active_device.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
        
        // Print active streams
        std::cout << "Active streams:" << std::endl;
        for (auto stream : profile.get_streams()) {
            auto stream_profile = stream.as<rs2::video_stream_profile>();
            std::cout << "  - " << stream.stream_name() << ": " 
                      << stream_profile.width() << "x" << stream_profile.height() 
                      << " @ " << stream_profile.fps() << "fps" << std::endl;
        }
        
        // Main loop - capture 30 frames
        std::cout << "\nCapturing frames... Press Ctrl+C to stop." << std::endl;
        for (int i = 0; i < 30; i++) {
            // Wait for frames
            rs2::frameset frames;
            try {
                frames = pipe.wait_for_frames(5000); // 5 second timeout
            } catch (const rs2::error& e) {
                std::cerr << "Error waiting for frames: " << e.what() << std::endl;
                continue;
            }
            
            // Process frames that are available
            std::cout << "Frame " << i << ":" << std::endl;
            
            // Try to get color frame (may not exist if only depth was enabled)
            try {
                if (frames.get_color_frame()) {
                    auto color_frame = frames.get_color_frame();
                    std::cout << "  Color: " << color_frame.get_width() << "x" << color_frame.get_height() << std::endl;
                }
            } catch (...) {
                // Color frame not available
            }
            
            // Try to get depth frame
            try {
                if (frames.get_depth_frame()) {
                    auto depth_frame = frames.get_depth_frame();
                    std::cout << "  Depth: " << depth_frame.get_width() << "x" << depth_frame.get_height() << std::endl;
                    
                    // Get center depth
                    float center_depth = depth_frame.get_distance(
                        depth_frame.get_width() / 2, 
                        depth_frame.get_height() / 2
                    );
                    std::cout << "  Center depth: " << center_depth << " meters" << std::endl;
                }
            } catch (...) {
                // Depth frame not available
            }
        }
        
        // Stop streaming
        pipe.stop();
        std::cout << "Pipeline stopped." << std::endl;
        
    } catch (const rs2::error& e) {
        std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n"
                  << "    " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}