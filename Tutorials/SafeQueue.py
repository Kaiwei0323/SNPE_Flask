import threading
from collections import deque

class SafeQueue:
    def __init__(self, max_size=25):
        self.queue = deque()
        self.max_size = max_size
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)

    def is_empty(self):
        with self.lock:
            return len(self.queue) == 0

    def is_full(self):
        with self.lock:
            return len(self.queue) == self.max_size

    def put(self, item):
        with self.not_full:
            while self.is_full():
                self.not_full.wait()  # Wait until the queue is not full
            self.queue.append(item)
            self.not_empty.notify()  # Notify that the queue is not empty now

    def get(self):
        with self.not_empty:
            while self.is_empty():
                self.not_empty.wait()  # Wait until the queue is not empty
            item = self.queue.popleft()  # FIFO behavior
            self.not_full.notify()  # Notify that the queue is not full now
            return item

