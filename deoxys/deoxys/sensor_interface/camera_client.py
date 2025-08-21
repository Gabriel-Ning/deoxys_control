import time
import cv2
import numpy as np
from deoxys.sensor_interface.network import ZMQCameraSubscriber
from deoxys.utils.cam_utils import notify_component_start

class CameraClient:
    def __init__(self, camera_info):
        self.camera_info = camera_info
        self.use_color = True
        self.use_depth = camera_info.cfg['depth']
        self.camera_id = camera_info.camera_id
        self.camera_name = camera_info.camera_name
        self.camera_type = camera_info.camera_type

        if self.use_color:
            self.color_sub = ZMQCameraSubscriber(
                    self.camera_info.cfg['host'], self.camera_info.cfg['port'] + self.camera_id, "RGB"
                )
        if self.use_depth and self.camera_type == "realsense":
            self.depth_sub = ZMQCameraSubscriber(
                self.camera_info.cfg['host'], self.camera_info.cfg['port'] + self.camera_info.cfg['depth_port_offset'] + self.camera_id, "Depth"
            )

    def start(self):
        notify_component_start(f"CameraClient for {self.camera_name}")

    def get_img(self):
        img_color = None
        img_depth = None
        if self.use_color:
            img_color, _ = self.color_sub.recv_rgb_image()
        if self.use_depth and self.camera_type == "realsense":
            img_depth, _ = self.depth_sub.recv_depth_image()
        return {"color": img_color, "depth": img_depth}

    def close(self):
        if self.use_color:
            self.color_sub.stop()
        if self.use_depth and self.camera_type == "realsense":
            self.depth_sub.stop()


if __name__ == "__main__":
    from deoxys.utils.cam_utils import load_camera_config
    camera_configs = load_camera_config()
    cam_client = CameraClient(camera_configs[0])
    cam_client.start()
    try:
        while True:
            imgs = cam_client.get_img()
            if imgs["color"] is not None:
                cv2.imshow("Color", imgs["color"])
            if imgs["depth"] is not None:
                cv2.imshow("Depth", imgs["depth"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam_client.close()
        cv2.destroyAllWindows()


    """
    This is the Python Interface for getting image name from redis server. 
    """

    def __init__(
        self,
        camera_info,
        host=None,
        port=None,
        use_color=True,
        use_depth=False
    ):
        self._host = host
        self._cam_port = port
        self._cam_configs = camera_info

        if self._host is None:
            self._host = camera_info.cfg['host'] if camera_info else "localhost"
        if self._cam_port is None:
            self._cam_port = camera_info.cfg['port'] if camera_info else 8000

        self.camera_id = camera_info.camera_id
        self.use_depth = use_depth

        self.camera_type = None

    def start(self, timeout=5):

        start_time = time.time()
        end_time = start_time
        while end_time - start_time < timeout:
            json_str = self.info_redis.get(f"{self.camera_name}::last_img_info")
            if json_str is not None:
                for _ in range(5):
                    self.save_img(flag=True)
                    time.sleep(0.02)

                img_info = self.get_img_info()
                self.camera_type = img_info["camera_type"]
                return True
            end_time = time.time()

        raise ValueError

    def stop(self):
        for _ in range(5):
            self.save_img(flag=False)
            time.sleep(0.02)

    def save_img(self, flag=False):
        if flag:
            self.info_redis.set(f"{self.camera_name}::save", 1)
        else:
            self.info_redis.delete(f"{self.camera_name}::save")

    def get_img_info(self):
        img_info = self.info_redis.get(f"{self.camera_name}::last_img_info")
        if img_info is not None:
            img_info = json.loads(img_info)
        return img_info

    def get_img(self):
        img_color = None
        img_depth = None
        if self.use_color:
            color_buffer = self.img_redis.get(f"{self.camera_name}::last_img_color")
            h, w, c = struct.unpack(">III", color_buffer[:12])
            img_color = np.frombuffer(color_buffer[12:], dtype=np.uint8).reshape(h, w, c)

        if self.use_depth:
            depth_buffer = self.img_redis.get(f"{self.camera_name}::last_img_depth")
            h, w = struct.unpack(">II", depth_buffer[:8])
            img_depth = np.frombuffer(depth_buffer[8:], dtype=np.uint16).reshape(h, w)
        return {"color": img_color, "depth": img_depth}

    def finish(self):
        for _ in range(10):
            self.save_img(flag=False)
            time.sleep(0.02)
        self.info_redis.set(f"{self.camera_name}::finish", 1)

    def close(self):
        self.finish()