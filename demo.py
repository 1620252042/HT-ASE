import argparse
from pathlib import Path

from PIL import Image
from psgan import Inference
from fire import Fire
import numpy as np
import os
import faceutils as futils
from psgan import PostProcess
from setup import setup_config, setup_argparser


import os

def log(log_info: str):

    print_log = True
    if print_log:
        print(log_info)


def read_source_images(source_path: str) -> list:

    paths = []
    log("Read source images path")
    if os.path.isdir(source_path):
        log("Source image path os OK!")
        for image_name in os.listdir(source_path):
            log(f"Get image name:{image_name}")
            image_path = os.path.join(source_path, image_name)
            paths.append(image_path)
    else:
        log("Error source image path")
    return paths

def get_image_name(image_path: str) -> str:

    log(f"Get {image_path}'s file name ...")
    folder_path = os.path.dirname(image_path)
    log(f"Get image's folder: {folder_path}")
    image_name = image_path.replace("{}/".format(folder_path), "")
    log(f"Image name is {image_name}")
    new_image_folder_name = str(image_name).split(".")[0]
    log(f"Folder name: {new_image_folder_name}")
    return new_image_folder_name


def get_image_name_with_end(image_path: str) -> str:

    log(f"Get {image_path}'s file name ...")
    folder_path = os.path.dirname(image_path)
    log(f"Get image's folder: {folder_path}")
    image_name = image_path.replace("{}/".format(folder_path), "")
    return image_name

def get_not_abs_image_name_with_end(image_path: str) -> str:

    image_name = image_path.split("/")[-1]
    log(f"Get {image_path}'s file name:{image_name}.")
    return image_name


def create_folder(folder_path: str) -> bool:

    create_state = False
    log(f"Create new image folder: {folder_path}")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        create_state = os.path.exists(folder_path)
        if create_state:
            log(f"Create folder: {folder_path} success!")
        else:
            log(f"Create folder: {folder_path} error!")
    else:
        create_state = True
        log(f"Folder {folder_path} allread exists!")
    return create_state


def copy_image(image_path: str, folder_path: str) -> bool:

    file_name = get_image_name_with_end(image_path=image_path)
    with open(image_path, 'rb') as rstream:
        container = rstream.read()
        new_image_path = os.path.join(folder_path, file_name)
        with open(new_image_path, 'wb') as wstream:
            wstream.write(container)
    if os.path.exists(new_image_path):
        log(f"Copy image from {image_path} to {new_image_path} success!")
        return True
    else:
        log(f"Copy image from {image_path} to {new_image_path} error!")
        return False


def create_style_image_name(source_image_name: str, style_image_name: str) -> str:

    # source image name: paly.png
    # style image name: omg.png
    # return paly_omg.png

    new_name = ''
    if "." in source_image_name:
        file_ends = source_image_name.split(".")[1]
        file_name = source_image_name.split(".")[0]
        style_name = style_image_name.split(".")[0]
        new_name = "{}_{}.{}".format(file_name, style_name, file_ends)
    return new_name


def main(save_path='./result/'):
    parser = setup_argparser()
    parser.add_argument(
        "--source_path",
        default="./assets/images/non-makeup",
        metavar="FILE",
        help="path to source image")
    parser.add_argument(
        "--reference_dir",
        default="assets/images/makeup",
        help="path to reference images")
    
    parser.add_argument(
        "--source_seg_path",
        default="./assets/segs/non-makeup",
        metavar="FILE",
        help="path to source image")
    parser.add_argument(
        "--reference_seg_path",
        default="assets/segs/makeup",
        help="path to reference images")
    
    parser.add_argument(
        "--speed",
        action="store_true",
        help="test speed")
    parser.add_argument(
        "--device",
        default="cpu",
        help="device used for inference")
    parser.add_argument(
        "--model_path",
        default="assets/models/G.pth",
        help="model for loading")

    args = parser.parse_args()
    config = setup_config(args)

    # Using the second cpu
    inference = Inference(config, args.device, args.model_path)
    postprocess = PostProcess(config)
    reference_paths = list(Path(args.reference_dir).glob("*"))
    #np.random.shuffle(reference_paths)
    source_paths = list(Path(args.source_path).glob("*"))
    #np.random.shuffle(source_paths)
    sou_num=0
    for source_path in source_paths:
        if not source_path.is_file():
            print(source_path, "is not a valid file.")
            continue
        source = Image.open(source_path).convert("RGB")
        print("-----------------------------------------")
        print("Read source image: {}".format(source_path))
        folder_name = get_not_abs_image_name_with_end(str(source_path)).split('.')[0]
        new_folder_root = os.path.join('/data/lmx/xiaorong/PSGAN6-30/result', folder_name)
        create_folder_states = create_folder(folder_path=new_folder_root)
        if create_folder_states:
            sou_num = sou_num + 1
            ref_num = 0
            for reference_path in reference_paths:
                print("Read style image: {}".format(reference_path))
                ref_num = ref_num + 1
                if not reference_path.is_file():
                    print(reference_path, "is not a valid file.")
                    
                    continue
                reference = Image.open(reference_path).convert("RGB")
                
                source_id=str(source_path).split('/')[-1]
                reference_id=str(reference_path).split('/')[-1]
                source_seg_path=os.path.join(args.source_seg_path, source_id)
                reference_seg_path=os.path.join(args.reference_seg_path, reference_id)
                
                
                
                source_seg = Image.open(source_seg_path).convert("RGB")
                reference_seg = Image.open(reference_seg_path).convert("RGB")
                
                print("----------------------------------------------")
                # Transfer the psgan from reference to source.
                image = inference.transfer(source, reference, source_seg, reference_seg)
                source_image_name = get_not_abs_image_name_with_end(source_id).split('.')[0]
                style_image_name = get_not_abs_image_name_with_end(reference_id).split('.')[0]
                new_image_path = "{}{}/{}_{}.png".format(save_path, source_image_name, source_image_name, style_image_name)
                log(f"Save image to {new_image_path}.")
                image.save(new_image_path)
                # import time
                # time.sleep(20)

                if args.speed:
                    import time
                    start = time.time()
                    for _ in range(100):
                        inference.transfer(source, reference)
                    print("Time cost for 100 iters: ", time.time() - start)





if __name__ == '__main__':
    main()
    # Get now path
    # now_path = '/data/lmx/xiaorong/PSGAN6-30'
    # new_folder_root = '/data/lmx/xiaorong/PSGAN6-30/result/'
    # source_path = os.path.join(now_path, "assets/images/non-makeup")
    # image_paths = read_source_images(source_path=source_path)
    # for item_path in image_paths:
    #     image_name = get_image_name(image_path=item_path)
    #     new_folder_path = os.path.join(new_folder_root, image_name)
    #     create_folder_state = create_folder(folder_path=new_folder_path)
    #     if create_folder_state:
    #         copy_source_image_state = copy_image(image_path=item_path, folder_path=new_folder_path)
