import time


class BenchmarkTimer:
    def __init__(self, unit="min"):
        self.timer_dic = []
        self.unit = unit
        if unit == "hr":
            self.unit_time = 3600.0
        elif unit == "min":
            self.unit_time = 60.0
        else:
            self.unit_time = 1

    def _calculate_time(self, start, end):
        return (end - start) / self.unit_time

    def start_step(self, name):
        self.timer_dic.append([name, time.time()])
        return "Start {}!".format(name)

    def end_step(self):
        end_time = time.time()
        self.timer_dic[-1].append(end_time)
        return "Finish {}! Step time elapsed: {:.2f} {}".format(
            self.timer_dic[-1][0], self._calculate_time(self.timer_dic[-1][1], end_time), self.unit
        )

    def breakdown(self):
        return ["{}: {:.2f} {}".format(item[0], self._calculate_time(item[1], item[2]), self.unit) for item in self.timer_dic]

    def total_time(self):
        return "{:.2f} {}".format(sum([self._calculate_time(item[1], item[2]) for item in self.timer_dic]), self.unit)


def get_elapsed_min(start_time):
    return "{:.2f} min".format((time.time() - start_time) / 60.0)