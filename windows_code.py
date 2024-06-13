import subprocess

def get_wmic_output(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout.strip().split('\n')[1].strip()

def get_system_id():
    # 获取各个硬件的识别码
    disk_serial_number = get_wmic_output("wmic diskdrive get serialnumber")
    bios_serial_number = get_wmic_output("wmic bios get serialnumber")
    cpu_id = get_wmic_output("wmic cpu get processorid")
    baseboard_serial_number = get_wmic_output("wmic baseboard get serialnumber")
    
    # 将所有识别码组合成一个唯一的系统ID
    system_id = f"{disk_serial_number}-{bios_serial_number}-{cpu_id}-{baseboard_serial_number}"
    return system_id

# 调用函数并打印结果
print(get_system_id())




# 第二种方式 


import subprocess
import hashlib
import platform
import uuid
import psutil
import datetime
import locale
import hashlib

def get_system_info():
    # 初始化系统信息字典
    info = {"cpu_model": platform.processor()}

    # 获取BIOS版本
    if platform.system() == "Windows":
        try:
            info["bios_version"] = subprocess.check_output(
                "wmic bios get smbiosbiosversion", shell=True
            ).decode().split("\n")[1].strip()
        except Exception as e:
            info["bios_version"] = str(e)
    elif platform.system() == "Darwin":
        try:
            info["bios_version"] = subprocess.check_output(
                "system_profiler SPHardwareDataType | grep 'Boot ROM Version'",
                shell=True,
            ).decode().split(": ")[1].strip()
        except Exception as e:
            info["bios_version"] = str(e)

    # 获取主板型号
    if platform.system() == "Windows":
        try:
            info["motherboard_model"] = subprocess.check_output(
                "wmic baseboard get product", shell=True
            ).decode().split("\n")[1].strip()
        except Exception as e:
            info["motherboard_model"] = str(e)
    elif platform.system() == "Darwin":
        try:
            info["motherboard_model"] = subprocess.check_output(
                "system_profiler SPHardwareDataType | grep 'Model Identifier'", shell=True,
            ).decode().split(": ")[1].strip()
        except Exception as e:
            info["motherboard_model"] = str(e)

    # 获取磁盘序列号
    if platform.system() == "Windows":
        try:
            info["disk_serial"] = subprocess.check_output(
                "wmic diskdrive get serialnumber", shell=True
            ).decode().split("\n")[1].strip()
        except Exception as e:
            info["disk_serial"] = str(e)
    elif platform.system() == "Darwin":
        try:
            info["disk_serial"] = subprocess.check_output(
                "system_profiler SPSerialATADataType | grep 'Serial Number'", shell=True,
            ).decode().split(": ")[1].strip()
        except Exception as e:
            info["disk_serial"] = str(e)

    # 获取显卡型号
    if platform.system() == "Windows":
        try:
            info["gpu_model"] = subprocess.check_output(
                "wmic path win32_videocontroller get name", shell=True
            ).decode().split("\n")[1].strip()
        except Exception as e:
            info["gpu_model"] = str(e)
    elif platform.system() == "Darwin":
        try:
            info["gpu_model"] = subprocess.check_output(
                "system_profiler SPDisplaysDataType | grep 'Chipset Model'", shell=True
            ).decode().split(": ")[1].strip()
        except Exception as e:
            info["gpu_model"] = str(e)

    # 获取内存大小、CPU核心数、操作系统版本、系统语言和当前日期
    info["memory_size"] = str(round(psutil.virtual_memory().total / (1024 ** 3))) + " GB"
    info["processor_count"] = str(psutil.cpu_count(logical=False))
    info["os_version"] = platform.version()
    info["system_language"] = locale.getlocale()[0]
    info["current_date"] = datetime.datetime.now().strftime("%Y-%m-%d")
    return info



lh = hashlib.md5()

system_info = get_system_info()
system_info_str = '|'.join([f"{key}:{value}" for key, value in system_info.items()])

lh.update(system_info_str.encode(encoding="utf-8"))
print(system_info_str)
print(lh.hexdigest())
