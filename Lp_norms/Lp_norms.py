"""
Lp_norms.py

Interactive demo with adjustable p, 
showing the shape of the L norm unit-balls
https://en.wikipedia.org/wiki/Lp_space

Mathematically L_inf = argmax(x_1, x_2, ... x_n), but
is visually approximated with L_300 by clicking the button.

Further reading/watching
MIT OpenCourseWare: 8. Norms of Vectors and Matrices
https://www.youtube.com/watch?v=NcPUI7aPFhA
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

init_p = 1
fig, ax = plt.subplots()
ax_p = plt.axes([0.15, 0.02, 0.30, 0.03])
slider_p = Slider(ax_p, 'p norm', 1e-1, 6, init_p)

ax_p_inf = plt.axes([0.55, 0.02, 0.2, 0.03])
btn_p_inf = Button(ax_p_inf, 'set p=300')

ax.set_aspect('equal')
manual_contour = [0.5, 1, 1.5]

x = np.arange(-1.2, 1.2, 0.01)
y = np.arange(-1.2, 1.2, 0.01)
xx, yy = np.meshgrid(x, y)


def update_slider(self):
    p = slider_p.val
    zz = (np.abs(xx) ** p + np.abs(yy) ** p) ** (1 / p)
    ax.clear()
    ax.set_title(f"Lp Norms. Contours at {manual_contour}", loc='left')
    ax.contourf(xx, yy, zz, 30, cmap='Spectral')
    ax.contour(xx, yy, zz, manual_contour, colors='blue', linewidths=2)


def update_with_p_inf(self):
    slider_p.set_val(300)
    update_slider(slider_p)


slider_p.on_changed(update_slider)
btn_p_inf.on_clicked(update_with_p_inf)

update_slider(slider_p)  # update manually to load data to display on startup
plt.ioff()
plt.show()
