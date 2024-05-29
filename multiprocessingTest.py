import multiprocessing
import time
import random

def task(n):
    while True:
        delay = random.uniform(0.5, 3.0)
        print(f"Task {n} will sleep for {delay:.2f} seconds.")
        time.sleep(delay)
        # return n * n

def init_process():
    # 如果需要在子进程启动时执行一些初始化，可以在这里定义
    pass

if __name__ == "__main__":
    # 创建一个包含4个进程的进程池
    pool = multiprocessing.Pool(processes=4, initializer=init_process)

    try:
        # 提交多个任务给进程池
        results = [pool.apply_async(task, args=(i,)) for i in range(10)]
        print("开始执行任务...")
        
        # 主进程等待3秒钟
        time.sleep(3)
        
        # 在3秒后终止进程池，立即终止所有子进程
        pool.terminate()  
        
        # 获取任务结果
        for result in results:
            try:
                # output = result.get(timeout=5)  # 设置超时时间为5秒
                output = result.get(timeout=0.1)  # 设置超时时间为5秒
                print(f"Result: {output}")
            except multiprocessing.TimeoutError:
                print("A task timed out.")
            except Exception as e:
                print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, terminating the pool...")
        pool.terminate()  # 立即终止所有子进程
    else:
        pool.close()  # 关闭进程池，不再接受新的任务
    finally:
        pool.join()  # 等待所有子进程退出
        print("所有子进程退出")

    print("All tasks completed or manually terminated.")
