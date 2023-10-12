import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

def emforce(q1,r1,q2,r2):
    distance = np.linalg.norm(r1-r2)
    F = (8.987552e+9)*q1*q2*pow(distance,-3)*(r1-r2)
    return F

class PointCharge:
    pass

#k = 8.987552e+9
k = 9.0e+9
x = 0.0
y = 1.0
vx = 0.0
vy = 0.0
q = 1.0

xa = -1.0
ya = 0.0
qa = 1.0e-5

xb = 1.0
yb = 0.0
qb = 1.0e-5

init = [x,y,vx,vy] #(x位置,y位置,x速度,y速度)の初期条件
t_span = [0.0,15.0]
t_eval = np.linspace(*t_span,3000)


def equation(t,X):
    x,y,vx,vy = X
    da = (x-1)**2+y**2
    da = da**1.5
    db = (x+1)**2+y**2
    db = db**1.5
    return [vx,vy,k*q*(qa*(x-1)/da-qb*(x+1)/db),k*q*y*(qa/da-qb/db)]

sol = solve_ivp(equation,t_span,init,method='RK45',t_eval=t_eval)
fig,ax = plt.subplots()
ax.set_xlim(-3.0,3.0)
ax.set_ylim(-3.0,3.0)
ax.set_aspect('equal')

ax.plot(xa,ya,'o')
ax.plot(xb,yb,'o')
point, = ax.plot([],[],'^')
def update_anim(frame_num):
    point.set_data(sol.y[0,frame_num],sol.y[1,frame_num]) 
    return point,

anim = FuncAnimation(fig,update_anim,frames=np.arange(0,len(t_eval)),interval=0.005,blit=True,repeat=True)
plt.show()
#print(sol)
