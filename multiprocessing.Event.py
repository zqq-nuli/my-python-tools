import multiprocessing
import time

# 全局标志
stop_B = multiprocessing.Event()
restart_B = multiprocessing.Event()
should_terminate = multiprocessing.Event()

def function_A():
    while True:
        # 触发 A 的相关逻辑
        print("Function A is running")
        time.sleep(2)

        # 设置停止 B 的标志
        stop_B.set()
        time.sleep(5)  # 模拟 A 的其他操作

        # 清除标志
        stop_B.clear()
        restart_B.set()  # 重新启动 B

def function_B():
    while not should_terminate.is_set():
        if not stop_B.is_set():
            # 触发 B 的相关逻辑
            print("Function B is running")
            for _ in range(10):  # 原本是 time.sleep(10)
                if stop_B.is_set() or should_terminate.is_set():
                    print("Function B is stopped")
                    break
                time.sleep(1)
        else:
            print("Function B is stopped")
            restart_B.wait()  # 等待重新启动信号
            restart_B.clear()
        time.sleep(1)

def function_C():
    while True:
        # 触发 C 的相关逻辑
        print("Function C is running")
        time.sleep(3)

        # 设置停止 B 的标志
        stop_B.set()
        time.sleep(5)  # 模拟 C 的其他操作

        # 清除标志
        stop_B.clear()
        restart_B.set()  # 重新启动 B

def terminate():
    should_terminate.set()
    restart_B.set()  # 确保 B 不再等待

if __name__ == "__main__":
    # 创建进程
    process_A = multiprocessing.Process(target=function_A)
    process_B = multiprocessing.Process(target=function_B)
    process_C = multiprocessing.Process(target=function_C)

    # 启动进程
    process_A.start()
    process_B.start()
    process_C.start()

    try:
        # 等待进程完成（在这种情况下，进程会一直运行，所以主进程会一直等待）
        process_A.join()
        process_B.join()
        process_C.join()
    except KeyboardInterrupt:
        # 捕获键盘中断信号，终止进程
        terminate()
        process_A.terminate()
        process_B.terminate()
        process_C.terminate()
        process_A.join()
        process_B.join()
        process_C.join()
