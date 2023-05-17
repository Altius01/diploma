import time

class Timing():
    time_stamps = {}
    elapsed_times = {}

    _time_interval = 0

    def add_timestamp(self, time_stamp_name:str):
        self.time_stamps[time_stamp_name] = self._time_interval = time.time()

    def add_elapsed_time_start(self):
        self._time_interval = time.time()

    def add_elapsed_time_end(self, time_stamp_name: str):
        self._time_interval = time.time() - self._time_interval
        a = self.elapsed_times.get(time_stamp_name, [0, 0])
        a[0] += self._time_interval
        a[1] += 1
        self.elapsed_times[time_stamp_name] = a
        self._time_interval = 0
