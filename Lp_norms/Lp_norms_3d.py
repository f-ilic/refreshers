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
from mpl_toolkits.mplot3d import Axes3D



init_p = 2
# fig, ax = plt.subplots(projection='3d')
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax_p = plt.axes([0.15, 0.02, 0.30, 0.03])
slider_p = Slider(ax_p, 'p norm', 1e-1, 6, init_p)

ax_p_inf = plt.axes([0.55, 0.02, 0.2, 0.03])
btn_p_inf = Button(ax_p_inf, 'set p=300')

ax.set_aspect('auto')
manual_contour = [0.25, 0.5, 0.75, 1]


x = np.arange(-1.1, 1.1, 0.04)
y = np.arange(-1.1, 1.1, 0.04)
z = np.arange(-1.1, 1.1, 0.04)
xx, yy, zz = np.meshgrid(x, y, z)



def update_slider(self):
    p = slider_p.val
    norm = (np.abs(xx) ** p + np.abs(yy) ** p + np.abs(zz)** p) ** (1 / p)
    ax.clear()    
    norm[(norm>1.0)] = np.NaN
    cmap = plt.get_cmap('Spectral')
    cmap.set_bad(color='white', alpha = 0)
    ax.scatter3D(xx, yy, zz, c=norm[...,], marker='.', s=20, cmap=cmap)



def update_with_p_inf(self):
    slider_p.set_val(300)
    update_slider(slider_p)


slider_p.on_changed(update_slider)
btn_p_inf.on_clicked(update_with_p_inf)

update_slider(slider_p)  # update manually to load data to display on startup
plt.ioff()
plt.show()
