import cv2
import numpy as np
import zmq
import os
import struct
from pyorbbecsdk import *
import open3d as o3d
import time
import traceback
from collections import deque

class OrbbecCamera:
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.context = Context()
        self.pipeline = None
        self.align_filter = None
        self.point_cloud_filter = None
        self.color_profile = None
        self.depth_profile = None
        self.device = None
        self.serial_number = None
        self.save_points_dir = os.path.join(os.getcwd(), "point_clouds")
        if not os.path.exists(self.save_points_dir):
            os.mkdir(self.save_points_dir)

    def get_camera_info(self):
        """Enumerate available devices and get their info"""
        device_list = self.context.query_devices()
        if device_list.get_count() < 1:
            print("No device found, please connect a device and try again.")
            return None, None

        devices_info = []
        print("\nEnumerated devices:")
        for index in range(device_list.get_count()):
            device = device_list[index]
            device_info = device.get_device_info()
            info = {
                'index': index,
                'name': device_info.get_name(),
                'pid': device_info.get_pid(),
                'serial_number': device_info.get_serial_number()
            }
            devices_info.append(info)
            print(f" - {index}. Device name: {info['name']}, "
                  f"PID: {info['pid']}, "
                  f"Serial Number: {info['serial_number']}")

        return devices_info, device_list

    def initialize(self, serial_number=None):
        """Initialize camera with optional serial number"""
        devices_info, device_list = self.get_camera_info()
        
        if not devices_info:
            return False

        try:
            if serial_number:
                self.device = device_list.get_device_by_serial_number(serial_number)
                self.serial_number = serial_number
            else:
                # Let user select device
                while True:
                    device_selected = self._get_input_option()
                    if device_selected == -1:
                        return False
                    if 0 <= device_selected < len(devices_info):
                        self.device = device_list.get_device_by_index(device_selected)
                        self.serial_number = devices_info[device_selected]['serial_number']
                        break
                    print("Invalid input, please select again!")

            return self._setup_pipeline()

        except Exception as e:
            print(f"Failed to initialize device: {e}")
            return False

    def _get_input_option(self):
        """Get user input for device selection"""
        option = input("Please select a device (or press 'q' to exit): ")
        if option.lower() == 'q':
            return -1
        try:
            return int(option)
        except ValueError:
            print("Invalid input, please enter a number!")
            return self._get_input_option()

    def _setup_pipeline(self):
        """Setup pipeline with selected device"""
        try:
            self.pipeline = Pipeline(self.device)
            config = Config()

            # Setup color stream
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            self.color_profile = profile_list.get_video_stream_profile(
                self.width, self.height, OBFormat.RGB, self.fps)
            config.enable_stream(self.color_profile)
            
            # Setup depth stream
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            self.depth_profile = profile_list.get_video_stream_profile(
                self.width, self.height, OBFormat.Y16, self.fps)
            config.enable_stream(self.depth_profile)
            
            # Enable frame sync and start pipeline
            self.pipeline.enable_frame_sync()
            self.pipeline.start(config)
            
            # Initialize filters
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            self.point_cloud_filter = PointCloudFilter()
            
            print(f"Camera initialized successfully with serial number: {self.serial_number}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False

    def get_frames(self):
        """Get aligned color and depth frames"""
        try:
            frames = self.pipeline.wait_for_frames(100)
            if not frames:
                return None, None, None

            # Get aligned frames
            aligned_frames = self.align_filter.process(frames)
            if not aligned_frames:
                return None, None, None

            frame_set = aligned_frames.as_frame_set()
            color_frame = frame_set.get_color_frame()
            depth_frame = frame_set.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None, None

            return color_frame, depth_frame, frame_set

        except Exception as e:
            print(f"Error getting frames: {e}")
            return None, None, None

    def frame_to_image(self, color_frame, depth_frame):
        """Convert frames to numpy arrays"""
        try:
            # Convert color frame to BGR image
            color_data = np.asanyarray(color_frame.get_data())
            color_image = np.resize(color_data, (self.height, self.width, 3))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # Convert depth frame to colorized depth map
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((self.height, self.width))
            depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()

            # Apply depth thresholds
            depth_data = np.where((depth_data > 20) & (depth_data < 3000), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)

            # Colorize depth map
            depth_colormap = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
            depth_colormap = cv2.applyColorMap(depth_colormap.astype(np.uint8), 
                                             cv2.COLORMAP_JET)

            return color_image, depth_colormap, depth_data

        except Exception as e:
            print(f"Error converting frames to images: {e}")
            return None, None, None

    def get_point_cloud(self, frame_set, has_color=True):
        """Generate point cloud from frame set"""
        try:
            if frame_set is None:
                raise ValueError("Frame set is None")

            point_format = OBFormat.RGB_POINT if has_color else OBFormat.POINT
            self.point_cloud_filter.set_create_point_format(point_format)
            
            point_cloud_frame = self.point_cloud_filter.process(frame_set)
            points = self.point_cloud_filter.calculate(point_cloud_frame)

            points_array = np.array([p[:3] for p in points])
            colors_array = np.array([p[3:6] for p in points]) if has_color else None

            return points_array, colors_array

        except Exception as e:
            print(f"Error generating point cloud: {e}")
            return None, None

    def save_point_cloud(self, points, colors, filename):
        """Save point cloud to PLY file"""
        if points is None or len(points) == 0:
            print("No points to save.")
            return False

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

            filepath = os.path.join(self.save_points_dir, filename)
            o3d.io.write_point_cloud(filepath, pcd)
            print(f"Point cloud saved to: {filepath}")
            return True

        except Exception as e:
            print(f"Error saving point cloud: {e}")
            return False

    def get_camera_parameters(self):
        """Get camera intrinsic and extrinsic parameters"""
        try:
            return {
                'depth_intrinsics': self.depth_profile.get_intrinsic(),
                'color_intrinsics': self.color_profile.get_intrinsic(),
                'depth_distortion': self.depth_profile.get_distortion(),
                'color_distortion': self.color_profile.get_distortion(),
                'extrinsic': self.depth_profile.get_extrinsic_to(self.color_profile)
            }
        except Exception as e:
            print(f"Error getting camera parameters: {e}")
            return None

    def release(self):
        """Stop pipeline and release resources"""
        if self.pipeline:
            self.pipeline.stop()

class RealSenseCamera:
    """Camera wrapper for Intel RealSense cameras"""
    
    def __init__(self, width=1280, height=720):
        """Initialize with desired resolution"""
        self.width = width
        self.height = height
        self.pipeline = None
        self.config = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the RealSense camera"""
        import pyrealsense2 as rs
        
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure streams
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
            
            # Start the pipeline
            profile = self.pipeline.start(self.config)
            
            # Get intrinsic parameters (if needed)
            self.color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            
            # Allow auto-exposure to stabilize
            for _ in range(30):
                self.pipeline.wait_for_frames()
                
            self.initialized = True
            print("RealSense camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize RealSense camera: {e}")
            traceback.print_exc()
            self.initialized = False
            return False
            
    def get_frames(self):
        """Get color and depth frames from the camera"""
        if not self.initialized:
            return None, None, None
            
        try:
            import pyrealsense2 as rs
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            return color_frame, depth_frame, frames
            
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None, None, None
            
    def frame_to_image(self, color_frame, depth_frame):
        """Convert frames to numpy arrays for processing"""
        if color_frame is None:
            return None, None, None
            
        try:
            # Get color image
            color_image = np.asanyarray(color_frame.get_data())
            
            # Get depth image
            depth_image = None
            depth_colormap = None
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
            return color_image, depth_image, depth_colormap
            
        except Exception as e:
            print(f"Error converting frames to images: {e}")
            return None, None, None
            
    def release(self):
        """Stop the camera pipeline"""
        if self.pipeline:
            try:
                import pyrealsense2 as rs
                self.pipeline.stop()
                print("RealSense camera released")
            except Exception as e:
                print(f"Error releasing RealSense camera: {e}")
                
        self.initialized = False

def get_available_camera(width=1280, height=720, camera_type="rs"):
    """Try to initialize either RealSense or Orbbec camera based on input parameter"""
    if camera_type == "rs":
        # Try RealSense first
        try:
            import pyrealsense2 as rs
            camera = RealSenseCamera(width=width, height=height)
            if camera.initialize():
                print("Using Intel RealSense camera")
                return camera
        except ImportError:
            print("Intel RealSense SDK not available")
        except Exception as e:
            print(f"Error initializing RealSense: {e}")
    
    elif camera_type == "orbbec":
        # Try Orbbec
        try:
            camera = OrbbecCamera(width=width, height=height)
            if camera.initialize():
                print("Using Orbbec camera")
                return camera
        except ImportError:
            print("Orbbec SDK not available")
        except Exception as e:
            print(f"Error initializing Orbbec: {e}")
    
    print("No suitable camera found")
    return None

def basic_usage():
    # Example usage
    camera = get_available_camera(width=1280, height=720, camera_type="orbbec")
    
    if not camera:
        print("Failed to initialize camera")
        return

    try:
        cv2.namedWindow("Color", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

        while True:
            color_frame, depth_frame, frame_set = camera.get_frames()
            if color_frame is None or depth_frame is None:
                time.sleep(0.1)
                continue

            color_image, depth_colormap, depth_data = camera.frame_to_image(color_frame, depth_frame)
            if color_image is None:
                time.sleep(0.1)
                continue

            cv2.imshow("Color", color_image)
            cv2.imshow("Depth", depth_colormap)

            key = cv2.waitKey(30)
            if key == ord('q'):
                break
            elif key == ord('p'):
                # Save point cloud
                points, colors = camera.get_point_cloud(frame_set)
                if points is not None:
                    camera.save_point_cloud(points, colors, f"point_cloud_{time.time()}.ply")

    finally:
        camera.release()
        cv2.destroyAllWindows()

class ImageServer:
    """Image server class for streaming camera frames"""
    def __init__(self, camera, port=5555, unit_test=False):
        self.camera = camera
        self.port = port
        self.Unit_Test = unit_test
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

        if self.Unit_Test:
            self._init_performance_metrics()

        print("[Image Server] Image server has started, waiting for client connections...")

    def _init_performance_metrics(self):
        self.frame_count = 0  # Total frames sent
        self.time_window = 1.0  # Time window for FPS calculation (in seconds)
        self.frame_times = deque()  # Timestamps of frames sent within the time window
        self.start_time = time.time()  # Start time of the streaming

    def _update_performance_metrics(self, current_time):
        # Add current time to frame times deque
        self.frame_times.append(current_time)
        # Remove timestamps outside the time window
        while self.frame_times and self.frame_times[0] < current_time - self.time_window:
            self.frame_times.popleft()
        # Increment frame count
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window
            print(f"[Image Server] Real-time FPS: {real_time_fps:.2f}, Total frames sent: {self.frame_count}, Elapsed time: {elapsed_time:.2f} sec")

    def _close(self):
        self.camera.release()
        self.socket.close()
        self.context.term()
        print("[Image Server] The server has been closed.")

    def send_process(self):
        try:
            while True:
                color_frame, depth_frame, _ = self.camera.get_frames()
                if color_frame is None:
                    print("[Image Server] Camera frame read error.")
                    continue

                color_image, _, _ = self.camera.frame_to_image(color_frame, depth_frame)
                if color_image is None:
                    print("[Image Server] Frame conversion error.")
                    continue

                ret, buffer = cv2.imencode('.jpg', color_image)
                if not ret:
                    print("[Image Server] Frame encoding failed.")
                    continue

                jpg_bytes = buffer.tobytes()

                if self.Unit_Test:
                    timestamp = time.time()
                    frame_id = self.frame_count
                    header = struct.pack('dI', timestamp, frame_id)  # 8-byte double, 4-byte unsigned int
                    message = header + jpg_bytes
                else:
                    message = jpg_bytes

                self.socket.send(message)

                if self.Unit_Test:
                    current_time = time.time()
                    self._update_performance_metrics(current_time)
                    self._print_performance_metrics(current_time)

        except KeyboardInterrupt:
            print("[Image Server] Interrupted by user.")
        finally:
            self._close()
    
def main():
    camera = get_available_camera(width=1280, height=720, camera_type="orbbec")
    
    if not camera:
        print("Failed to initialize camera")
        return
    
    ImageServer(camera, port=5555, unit_test=True).send_process()
    camera.release()
    
if __name__ == "__main__":
    # Example usage
    # basic_usage()

    main()
