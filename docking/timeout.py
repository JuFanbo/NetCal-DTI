import threading
import traceback
def run_with_timeout(func, args=(), kwargs={}, timeout=10):
    class InterruptableThread(threading.Thread):
        def __init__(self):
            super().__init__()
            self.result = None
            self.exception = None
            self.traceback = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except Exception as e:
                self.exception = e
                self.traceback = traceback.format_exc()

    # 创建并启动线程
    thread = InterruptableThread()
    thread.start()

    # 等待线程完成，但最多等待timeout秒
    thread.join(timeout)

    if thread.is_alive():
        # 如果线程仍然在运行，说明超时了
        print(f"函数执行超时（{timeout}秒）")
        return None
    else:
        # 检查是否有异常
        if thread.exception:
            print(f"函数执行出错: {thread.exception}")
            print(thread.traceback)
            raise thread.exception
        return thread.result