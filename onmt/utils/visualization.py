from visdom import Visdom
import numpy as np



class Visdata():
    viz = Visdom(env='s', port=8099)
    x, y = 0, 0
    win = viz.line(
        X=np.array([x]),
        Y=np.array([y]),
        opts=dict(title='two_lines'))
    for i in range(10):
        x += i
        y += i
        viz.line(
            X=np.array([x]),
            Y=np.array([y]),
            win=win,  # win要保持一致
            update='append')

Visdata()