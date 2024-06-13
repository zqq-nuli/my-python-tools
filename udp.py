# pip install scapy

from scapy.all import sniff, UDP, IP
import logging

# 配置日志记录
logging.basicConfig(
    filename='udp_packets.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def udp_packet_callback(packet):
    if packet.haslayer(UDP):
        udp_layer = packet.getlayer(UDP)
        ip_layer = packet.getlayer(IP)
        
        # 检查数据包是否满足指定条件
        if ip_layer.src == '192.168.1.100' and udp_layer.sport == 12345:
            log_message = f"UDP Packet: {ip_layer.src}:{udp_layer.sport} -> {ip_layer.dst}:{udp_layer.dport} | Payload: {udp_layer.payload}"
            print(log_message)
            logging.info(log_message)

# 监听所有接口上的 UDP 数据包
sniff(filter="udp", prn=udp_packet_callback, store=0)




# to windows file 
# pip install paramiko
import paramiko
import os

# B 电脑的详细信息
hostname = "B_computer_ip"
port = 22
username = "your_username"
password = "your_password"

# 监听脚本的路径
remote_script_path = "/path/to/your/udp_sniff_script.py"

# 创建 SSH 客户端
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# 连接到 B 电脑
ssh_client.connect(hostname, port, username, password)

# 运行监听脚本
stdin, stdout, stderr = ssh_client.exec_command(f"python3 {remote_script_path}")

# 打印监听脚本的输出
for line in iter(stdout.readline, ""):
    print(line, end="")

# 传输日志文件到 A 电脑
sftp = ssh_client.open_sftp()
sftp.get('/path/to/udp_packets.log', 'udp_packets.log')
sftp.close()

# 关闭连接
ssh_client.close()
