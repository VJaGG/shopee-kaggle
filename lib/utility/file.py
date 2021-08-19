import os
import sys
import builtins


def open(file, mode=None, encoding=None):
    if mode == None:
        mode = 'r'
    if '/' in file:
        if 'w' or 'a' in mode:
            dir = os.path.dirname(file)
            if not os.path.isdir(dir):
                os.makedirs(dir)
    f = builtins.open(file, mode=mode, encoding=encoding)
    return f


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None
    
    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)
    
    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0
        
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        
        if is_file == 1:
            self.file.write(message)
            self.file.flush()
    
    def flush(self):
        pass


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t / 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    
    else:
        raise NotImplementedError