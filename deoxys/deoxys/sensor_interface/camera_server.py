import pyrealsense2 as rs
import time
import threading

from deoxys.sensor_interface.sensors.realsense import RealsenseCamera
from deoxys.sensor_interface.sensors.fisheye_cam import FishEyeCamera
from deoxys.utils.cam_utils import load_camera_config, CameraInfo

class CameraServer:
    def __init__(self, host: str, cam_port: int, cam_configs: list):
        self._host = host
        self._cam_port = cam_port
        self._cam_configs = cam_configs
        self._cam_threads = []

        if self._host is None:
            self._host = cam_configs[0].cfg['host'] if cam_configs else "localhost"
        if self._cam_port is None:
            self._cam_port = cam_configs[0].cfg['port'] if cam_configs else 8000

        for cam_info in self._cam_configs:
            if cam_info.camera_type == "realsense":
                ctx = rs.context()
                devices = ctx.query_devices()
                for dev in devices:
                    dev.hardware_reset()
                print("Waiting for hardware reset on cameras for 15 seconds...")
                time.sleep(1)
                break

    def _start_component(self, cam_config):
        cam_type = cam_config.camera_type
        cam_idx = cam_config.camera_id
        print(f"Starting camera {cam_idx} of type {cam_type}")
        if cam_type == "realsense":
            component = RealsenseCamera(
                host=self._host,
                port=self._cam_port + cam_idx,
                cam_id=cam_idx,
                cam_config=cam_config,
            )
        elif cam_type == "fisheye":
            component = FishEyeCamera(
                host=self._host,
                port=self._cam_port + cam_idx,
                cam_id=cam_idx,
                cam_config=cam_config,
            )
        else:
            raise ValueError(f"Invalid camera type: {cam_type}")
        component.stream()

    def _init_camera_threads(self):
        for cam_cfg in self._cam_configs:
            cam_thread = threading.Thread(
                target=self._start_component,
                args=(cam_cfg,),
                daemon=True,
            )
            cam_thread.start()
            self._cam_threads.append(cam_thread)

        for cam_thread in self._cam_threads:
            cam_thread.join()

if __name__ == "__main__":
    try:
        camera_configs = load_camera_config()
        camera_server = CameraServer(
            host=None,
            cam_port=None,
            cam_configs=camera_configs
        )
        camera_server._init_camera_threads()
        
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Exiting gracefully...")
        import sys
        sys.exit(0)
