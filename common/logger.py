class Logger():
    _logs = []

    def log(log_message, log_class="", log_method=""):
        _msg = LogMSG(log_message, log_class, log_method)
        # self._logs.append(_msg)
        _msg.log_print()


class LogMSG():
    log_class = None
    log_method = None
    log_message = None

    def __init__(self, log_message, log_class, log_method):
        self.log_class = log_class
        self.log_method = log_method
        self.log_message = log_message

    def log_print(self, file=None):
        print(f"LOG | {self.log_class}.{self.log_method}: {self.log_message}")
