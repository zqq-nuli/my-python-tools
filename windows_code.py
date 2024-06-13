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
