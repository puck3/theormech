import math

import numpy as n
import matplotlib.pyplot as plot
from matplotlib.animation import FuncAnimation

fgr = plot.figure()
gr = fgr.add_subplot(1, 1, 1)
gr.axis('equal')

step = 500
t = n.linspace(0, 2*math.pi, step)

Xo = 3
Yo = 4
R = 0.5
y0 = 2.3
r = 0.1
y = n.sin(t)

gr.plot([2, 4], [0, 0], 'black', linewidth=3)
gr.plot([2, 4], [Yo+0.7, Yo+0.7], 'black', linewidth=3)
gr.plot([Xo-0.1, Xo, Xo+0.1], [Yo+0.7, Yo, Yo+0.7], 'black')

y_l = y + y0
y_r = y - y0
phi = n.pi/18 * n.sin(2*t)

Xb = Xo - R
Yb = Yo

Xa = Xb + y_l * n.sin(phi)
Ya = Yb - y_l * n.cos(phi)

AB = gr.plot([Xa[0], Xb], [Ya[0], Yb], 'green')[0]
L = gr.plot([Xo + R, Xo + R], [Yo, Yo + y_r[0]], 'green')[0]

Alp = n.linspace(0, 2*n.pi, 100)
Xc = n.cos(Alp)
Yc = n.sin(Alp)

Block = gr.plot(Xo + R * Xc, Yo + R * Yc, 'black')[0]

m = gr.plot(Xa[0] + r * Xc, Ya[0] + r * Yc, 'black')[0]

Np = 20
Yp = n.linspace(0, 1, 2*Np+1)
Xp = 0.15 * n.sin(n.pi/2*n.arange(2*Np+1))
Pruzh = gr.plot(Xo + R + Xp, (Yo + y_r[0]) * Yp)[0]


def run(i):
    m.set_data([Xa[i] + r * Xc], [Ya[i] + r * Yc])
    AB.set_data([Xa[i], Xb], [Ya[i], Yb])
    L.set_data([Xo + R, Xo + R], [Yo, Yo + y_r[i]])
    Pruzh.set_data(Xo + R + Xp, (Yo + y_r[i]) * Yp)
    return [m, AB, Block, Pruzh]


anim = FuncAnimation(fgr, run, frames=step, interval=1)

plot.show()
