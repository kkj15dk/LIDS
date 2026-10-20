import time


class print_time:
    '''
    Lines are prefixed with [TIMING] to get an overview of which steps are slow.
    '''

    def __init__(self, desc: str):
        self.desc = desc

    def __enter__(self):
        print(f"[TIMING] {self.desc}...", flush=True)
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_seconds = int(round(time.time() - self._start))
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"[TIMING] {self.desc} took {hours:02d}:{minutes:02d}:{seconds:02d}", flush=True)
