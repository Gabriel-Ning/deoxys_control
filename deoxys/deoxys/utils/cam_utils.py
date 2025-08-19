import os
import yaml
from deoxys import config_root
import cv2

class RsCameraInfo:
    def __init__(self, camera_type, camera_id, camera_name):
        self.camera_type = camera_type
        self.camera_id = str(camera_id)
        self.camera_name = camera_name

    def __repr__(self):
        return f"RsCameraInfo(type={self.camera_type}, id={self.camera_id}, name={self.camera_name})"

def load_camera_config(yaml_path=None):
    if yaml_path is None:
        yaml_path = os.path.join(config_root, "camera_setup_config.yml")
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    camera_infos = []
    for entry in config["cameras"]:
        info = entry["camera_info"]
        cam = RsCameraInfo(
            camera_type=info["type"],
            camera_id=info["id"],
            camera_name=info["name"]
        )
        camera_infos.append(cam)
    return camera_infos

def resize_img(img, camera_type, img_w=128, img_h=128, offset_w=0, offset_h=0):

    if camera_type == "k4a":
        resized_img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
        w = resized_img.shape[0]
        h = resized_img.shape[1]

    if camera_type == "rs":
        resized_img = cv2.resize(img, (0, 0), fx=0.2, fy=0.3)
        w = resized_img.shape[0]
        h = resized_img.shape[1]

    resized_img = resized_img[
        w // 2 - img_w // 2 : w // 2 + img_w // 2,
        h // 2 - img_h // 2 : h // 2 + img_h // 2,
        :,
    ]
    return resized_img
