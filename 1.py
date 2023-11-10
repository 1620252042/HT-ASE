from PIL import Image
import os
import cv2
import numpy

def print_log(info: str, level:str="DEBUG", show: bool=True):

    show = True
    if level in ["DEBUG"]:
        if show:
            print(info)

def image_resize(image, width: int=256, height: int=256):

    print_log("对图片进行个性化处理...")
    print_log("重新设置图片大小为[width:{}, height:{}]".format(str(width), str(height)))
    img = cv2.resize(image, (width, height))
    return img

def save_image(image_path: str, image):

    print_log("开始保存图片{}".format(image_path))
    cv2.imwrite(image_path, image)

def get_target_folder_paths(target_folder_path: str) -> list:

    print_log(info="获取图片文件夹")
    folder_paths = []
    if not os.path.exists(target_folder_path):
        print_log(info="图片文件夹路径不存在")
    else:
        for item in os.listdir(target_folder_path):
            temp_path = os.path.join(target_folder_path, item)
            if not os.path.isdir(temp_path):
                print_log(info=f"{str(item)}非文件夹，跳过。")
            else:
                print_log(info=f"获取文件夹: {str(temp_path)}")
                temp_map = {"folder_name": item, "folder_path": temp_path}
                folder_paths.append(temp_map)
    return folder_paths

def check_tatget_file(file_name: str, file_ends: list=None) -> bool:

    result = False
    if not file_ends:
        result = True
    else:
        for item in file_ends:
            if str(file_name).endswith(item):
                result = True
                break
    print_log(info=f"Check file {file_name} in {str(file_ends)}:{str(result)}", level="INFO")
    return result

def get_image_info(source_folder_path: str, file_ends: list=None) -> list:

    print_log(info="获取源文件图片")
    image_names = []
    if not os.path.exists(source_folder_path):
        print_log(info="源图片文件夹路径不存在")
    else:
        for item in os.listdir(source_folder_path):
            print_log(info=f"获取文件: {str(item)}")
            if not os.path.isfile(os.path.join(source_folder_path, item)):
                print_log(info=f"{str(item)}非文件，跳过。")
                continue
            if check_tatget_file(file_name=item, file_ends=file_ends):
                image_info = {"image_name": item, "image_path": os.path.join(source_folder_path, item)}
                image_names.append(image_info)
    return image_names


def resize_image(image_path: str, image_name: str, folder_name: str, target_path: str) -> bool:
    print_log("开始读取:{image_path}.")
    temp_image = cv2.imread(image_path)
    # print_log("该图片的大小为:" + str(temp_image.shape))
    temp_image = image_resize(temp_image, 256, 256)
    output_image_path = os.path.join(target_path, f"{folder_name}_{image_name}")
    print_log(f"整合输出文件:{output_image_path}")
    save_image(image_path=output_image_path, image=temp_image)
    return os.path.isfile(output_image_path)


# 统一文件夹路径
folder_path = "./result"
# 输出文件夹
target_path = "./csat"
# 图片后缀
image_ends = ['.png', '.png']

folder_map = get_target_folder_paths(target_folder_path=folder_path)
for item in folder_map:
    if isinstance(item, dict):
        folder_name = str(item.get('folder_name'))
        folder_path = str(item.get('folder_path'))
        print_log(f"Folder name: {folder_name}")
        print_log(f"Folder path: {folder_path}")
        image_infos = get_image_info(source_folder_path=folder_path, file_ends=image_ends)
        for image_item in image_infos:
            if isinstance(image_item, dict):
                image_name = image_item.get("image_name")
                image_path = image_item.get("image_path")
                print_log(f"Image name: {image_name}")
                print_log(f"Image path: {image_path}")

                # create_new_data
                image_data = {"folder_name": folder_name, "image_name": image_name, "image_path": image_path}
                print(image_data)
                a = resize_image(image_path=image_path, image_name=image_name, folder_name=folder_name,
                                 target_path=target_path)
                print(a)
