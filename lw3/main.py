import numpy as n
import matplotlib.pyplot as plot
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint


def sys_diff_eq(y, t, m, M, l0, c, g, R):
    # y = [x, phi, x', phi'] -> dy = [x', phi', x'', phi'']
    dy = n.zeros_like(y)
    dy[0] = y[2]
    dy[1] = y[3]

    L = l0 + y[0] - R * y[1]
    # a11 * x'' + a12 * phi'' = b1
    # a21 * x'' + a22 * phi'' = b2

    a11 = M/2 + m
    a12 = 0
    b1 = m * g * n.cos(y[1]) - c * (y[0] + (m * g) / c) + m * L * y[3] * y[3]

    a21 = 0
    a22 = L
    b2 = -g * n.sin(y[1]) - y[3] * (2 * y[2] - R * y[3])

    det_a = a11 * a22 - a12 * a21
    det_a1 = b1 * a22 - a12 * b2
    det_a2 = a11 * b2 - a21 * b1

    dy[2] = det_a1 / det_a
    dy[3] = det_a2 / det_a

    return dy


step = 1000
t = n.linspace(0, 10, step)

y0 = [0.02, n.pi/6, 0, 0]

M = 1
m = 0.1
R = 0.4
l0 = 1
c = 50
g = 9.81

Y = odeint(sys_diff_eq, y0, t, (m, M, l0, c, g, R))

x = Y[:, 0]
phi = Y[:, 1]
x_t = Y[:, 2]
phi_t = Y[:, 3]

L = n.zeros_like(t)
x_tt = n.zeros_like(t)
phi_tt = n.zeros_like(t)
N_eps = n.zeros_like(t)
N_ita = n.zeros_like(t)


for i in range(len(t)):
    L[i] = l0 + x[i] + R * phi[i]
    x_tt[i] = sys_diff_eq(Y[i], t[i], m, M, L[i], c, g, R)[2]
    phi_tt[i] = sys_diff_eq(Y[i], t[i], m, M, L[i], c, g, R)[3]
    N_eps[i] = -m*(L[i]*phi_tt[i] + R*phi_t[i]**2 + 2*(x_t[i] - R*phi_t[i])*phi_t[i])*n.cos(phi[i]) - m*(x_tt[i] - L[i]*phi_t[i]**2)*n.sin(phi[i])
    N_ita[i] = -m*(L[i]*phi_tt[i] + R*phi_t[i]**2 + 2*(x_t[i] - R*phi_t[i])*phi_t[i])*n.sin(phi[i]) + m*(x_tt[i] - L[i]*phi_t[i]**2)*n.cos(phi[i]) - c*x[i] - (M + m) * g


fgr = plot.figure()
gr = fgr.add_subplot(4, 2, (1, 7))
gr.axis('equal')

x_plt = fgr.add_subplot(4, 2, 2)
x_plt.plot(t, x)
x_plt.set_title('x(t)')

phi_plt = fgr.add_subplot(4, 2, 4)
phi_plt.plot(t, phi)
phi_plt.set_title('phi(t)')

n_eps_plt = fgr.add_subplot(4, 2, 6)
n_eps_plt.plot(t, N_eps)
n_eps_plt.set_title('N_eps(t)')

n_ita_plt = fgr.add_subplot(4, 2, 8)
n_ita_plt.plot(t, N_ita)
n_ita_plt.set_title('N_ita(t)')

Yo = 1.4
r = 0.1
h_p0 = 0.7

Xb = -R
Yb = Yo

Xa = Xb + L * n.sin(phi)
Ya = Yb - L * n.cos(phi)

gr.plot([-2 * R, 2 * R], [0, 0], 'black', linewidth=3)
gr.plot([-2 * R, 2 * R], [Yo+0.7, Yo+0.7], 'black', linewidth=3)
gr.plot([-0.1, 0, 0.1], [Yo+0.7, Yo, Yo+0.7], 'black')

AB = gr.plot([Xa[0], Xb], [Ya[0], Yb], 'green')[0]
String_r = gr.plot([R, R], [Yo, h_p0 + x[0]], 'green')[0]

Alp = n.linspace(0, 2*n.pi, 100)
Xc = n.cos(Alp)
Yc = n.sin(Alp)

Block = gr.plot(R * Xc, Yo + R * Yc, 'black')[0]

Ticker = gr.plot(Xa[0] + r * Xc, Ya[0] + r * Yc, 'black')[0]

Np = 20
Yp = n.linspace(0, 1, 2*Np+1)
Xp = 0.05 * n.sin(n.pi/2*n.arange(2*Np+1))
Spring = gr.plot(R + Xp, (h_p0 + x[0]) * Yp)[0]


def run(i):
    Ticker.set_data([Xa[i] + r * Xc], [Ya[i] + r * Yc])
    AB.set_data([Xa[i], Xb], [Ya[i], Yb])
    String_r.set_data([R, R], [Yo, h_p0 + x[i]])
    Spring.set_data(R + Xp, (h_p0 + x[i]) * Yp)
    return [Ticker, AB, String_r, Spring]


anim = FuncAnimation(fgr, run, frames=step, interval=1)

plot.show()
