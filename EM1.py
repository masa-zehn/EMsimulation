import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

def emforce(q1,r1,q2,r2):
    distance = np.linalg.norm(r1-r2)
    F = (8.987552e+9)*q1*q2*pow(distance,-3)*(r1-r2)
    return F

class charge:
    pass

#k = 8.987552e+9
k = 9.0e+9
init = [0.0,1.0,0.0,0.0] #(x位置,y位置,x速度,y速度)の初期条件
t_span = [0.0,15.0]
t_eval = np.linspace(*t_span,3000)

def equationpv(pv,t):
    ret = [pv[1],emforce(1e-5,pv[0],1e-5,1+0j)+emforce(1e-5,pv[0],-1e-5,-1+0j)]
    return ret

def equation(t,X):
    x,y,vx,vy = X
    da = (x-1)**2+y**2
    da = da**1.5
    db = (x+1)**2+y**2
    db = db**1.5
    return [vx,vy,k*(1e-10)*((x-1)/da-(x+1)/db),k*(1e-10)*y*(1/da-1/db)]

sol = solve_ivp(equation,t_span,init,method='RK45',t_eval=t_eval)
fig,ax = plt.subplots()
ax.set_xlim(-3.0,3.0)
ax.set_ylim(-3.0,3.0)
ax.set_aspect('equal')

ax.plot(-1.0,0.0,'o')
ax.plot(1.0,0.0,'o')
point, = ax.plot([],[],'^')
def update_anim(frame_num):
    point.set_data(sol.y[0,frame_num],sol.y[1,frame_num]) 
    return point,

anim = FuncAnimation(fig,update_anim,frames=np.arange(0,len(t_eval)),interval=0.005,blit=True,repeat=True)
plt.show()
#print(sol)
