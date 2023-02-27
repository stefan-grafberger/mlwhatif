"""A simple way to monitor memory usage"""
import os
from time import sleep
import psutil


class MemoryMonitor:
    """A simple class to monitor memory usage"""
    # pylint: disable=too-few-public-methods

    def __init__(self):
        self.keep_measuring = True

    def measure_usage(self):
        """Measure memory usage using psutil"""
        usage_per_sec_psutil_rss = []
        usage_per_sec_psutil_vms = []
        process = psutil.Process(os.getpid())
        count = 0
        tmp_usage_per_sec_psutil_rss = []
        tmp_usage_per_sec_psutil_vms = []

        while self.keep_measuring:
            tmp_usage_per_sec_psutil_rss.append(process.memory_info().rss)
            tmp_usage_per_sec_psutil_vms.append(process.memory_info().vms)
            count += 1
            count = count % 10
            if count == 0:
                usage_per_sec_psutil_rss.append(max(tmp_usage_per_sec_psutil_rss))
                usage_per_sec_psutil_vms.append(max(tmp_usage_per_sec_psutil_vms))
                tmp_usage_per_sec_psutil_rss = []
                tmp_usage_per_sec_psutil_vms = []
            sleep(.1)
        # print(f"max_usage_per_sec_psutil_rss_max: {max(usage_per_sec_psutil_rss)}")
        # print(f"max_usage_per_sec_psutil_vms_max: {max(usage_per_sec_psutil_vms)}")
        return usage_per_sec_psutil_rss, usage_per_sec_psutil_vms
