import os
import sys
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def resource_path(relative_path: str) -> str:
    """
    获取资源文件的绝对路径，兼容开发环境和 PyInstaller 打包环境。

    Args:
        relative_path (str): 相对路径。

    Returns:
        str: 绝对路径。
    """
    if hasattr(sys, '_MEIPASS'):
        # 如果程序是打包后的，使用临时目录路径
        base_path = sys._MEIPASS
    else:
        # 在开发环境中，使用脚本所在的目录
        base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, relative_path)
    logging.debug(f'Resource path: {full_path}')
    return full_path

def find__img_list(autoPath: str) -> list:
    """
    获取给定文件夹下的所有图片的绝对路径。

    Args:
        autoPath (str): 文件夹路径。

    Returns:
        list: 包含所有图片绝对路径的列表。
    """
    # 使用 resource_path 函数处理路径
    dir_path = resource_path(autoPath)
    try:
        # 获取文件夹下的所有图片
        imgs = os.listdir(dir_path)
        # 获取图片的绝对路径
        imgs = [os.path.join(dir_path, img) for img in imgs]
        logging.debug(f'Image list: {imgs}')
        return imgs
    except FileNotFoundError as e:
        logging.error(f'FileNotFoundError: {e}')
        return []

def find_imgs(autoPath: str) -> tuple:
    """
    在给定的autoPath中查找图片列表，并在窗口中找到这些图片的中心点。

    Args:
        autoPath (str): 图片路径。

    Returns:
        tuple: 包含图片中心点的元组。
    """
    imgs = find__img_list(autoPath)
    # 假设 finder.find_images_all 是你自定义的函数，需要找到这些图片的中心点
    center_point = finder.find_images_all(imgs)
    return center_point

# 示例调用
if __name__ == "__main__":
    positions = find_imgs("public/death")
    if not positions:
        print("No images found.")
    else:
        print("Images found:", positions)
