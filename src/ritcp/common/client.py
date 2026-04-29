"""
Represents a high-level stream of images that could be fed into the incremental SfM and detection pipeline.
"""

# todo: dynamic clients move in the env (cars, ppl, etc)
# todo: static clients are fixed position in the map: fixed infrastructure cameras

from typing import List
import os

from src.ritcp.common.pose import Pose

class Client:
    """
    There could be multiple streams of images being sent to the map (multiple cars, multiple images oer vehicle, etc).
    Each ImageStream should have an associated location which can be used to optimize course-grain image matching in
    incremental SfM. Since there needs to be a correspondence (new image -> estimated location), each stream's data
    is maintained in a separate image directory.

    List of images populated from a unique image dir.
    Pose in the SfM map.
    Cursor/offset into the image list.
    """
    def __init__(self, images_path):
        self.images_path = images_path # the path to images
        # self.camera_info = None # colmap likes camera params to be known... could provide here for quick access?
        self.pose: Pose | None = None # current (tvec, qvec) in map coords for whatever device this stream comes from
        self.cursor = 0
        self.images: List[str] = Client._load_images(images_path)

    @staticmethod
    def _load_images(image_dir):
        extensions = ['jpg', 'jpeg', 'png']
        return [x for x in sorted(os.listdir(image_dir)) if x.lower().split('.')[-1] in extensions]

class StaticClient(Client):
    pass

class DynamicClient(Client):
    pass