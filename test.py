import mmcv

timer = mmcv.Timer()


from avcv import *
print(timer.since_last_check())