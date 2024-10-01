from threading import RLock


class Singleton(type):
    _instances = {}
    _lock = RLock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._lock.acquire()
            try:
                if cls not in cls._instances:
                    cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
            except Exception as e:
                raise e
            finally:
                cls._lock.release()
        return cls._instances[cls]

