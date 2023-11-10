import pickle
from pathlib import Path

import numpy as np
from PIL import Image


class DataReader:
    image_dir_name = "images"
    seg_dir_name = "segs"
    landmark_dir_name = "landmarks"
    makeup = "makeup.txt"
    non_makeup = "non-makeup.txt"

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir.joinpath(self.image_dir_name)
        self.seg_dir = self.data_dir.joinpath(self.seg_dir_name)
        self.lms_dir = self.data_dir.joinpath(self.landmark_dir_name)
        makeup_names = [name.strip() for name in self.data_dir.joinpath(self.makeup).open("rt")]
        non_makeup_names = [name.strip() for name in self.data_dir.joinpath(self.non_makeup).open("rt")]
        self.names = makeup_names + non_makeup_names

    def check_file(self, name):
        image = Image.open(
            self.image_dir.joinpath(name).as_posix()
        ).convert("RGB")
        seg = np.asarray(
            Image.open(
                self.seg_dir.joinpath(name).as_posix()
            )
        )
        lm = pickle.load(self.lms_dir.joinpath(name).open("rb"))

        try:
            assert lm.max() < min(image.size)-1
        except AssertionError:
            print(name)

    def check(self):
        for name in self.names:
            self.check_file(name)


if __name__ == '__main__':
    dr = DataReader()
    dr.check()


# makeup/765e7fbbd580c734d4469aaab60e1c8e.png
# makeup/32-41039.png
# makeup/caacbe9d52ffce79cb2d37469975583f.png
# makeup/vFG322.png
# makeup/vFG324.png
# makeup/vFG409.png
# makeup/vHX594.png
# makeup/vHX554.png
# makeup/vHX91.png
# makeup/vRX799.png