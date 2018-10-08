#! /usr/bin/env python

import sys
import os
from my_py_utils import interpolate_from_file

path = os.path.abspath(os.path.join(os.path.abspath(
       os.path.join(os.path.dirname(__file__), "..")), "python"))
sys.path.append(path)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import python.system_model
import python.mpc
import python.plotting

np.set_printoptions(suppress=True)

foot_l = []
foot_r = []

def interpolate(x1, y1, x2, y2):
   x_val = []
   y_val = []
   n = 140
   for i in range(1, n):
       a = float(i) / n             # rescale 0 < i < n --> 0 < a < 1
       x = (1 - a) * x1 + a * x2    # interpolate x coordinate
       y = (1 - a) * y1 + a * y2    # interpolate y coordinate
       x_val.append(x)
       y_val.append(y)
   x_val.append(x2)
   y_val.append(y2)

   return x_val, y_val

def main():

   #simulation constants
   dt           = 0.1
   t_step       = 0.8
   future_steps = 2
   #robot constants
   h_CoM        = 0.78 # 0.75
   foot_length  = 0.144
   foot_width   = 0.05
   h_step       = 0.05
   feet         = [foot_length, foot_width]

   #instantiate the linear system model
   #model = python.system_model.SystemModelDCM(h_CoM)
   model = python.system_model.SystemModel(h_CoM)

   #build the time vector
   # time_sim should be multiples of 0.8s
   time_sim = 8.0
   time     = np.arange(0, time_sim, dt)
   ntime    = time.size

   #instantiate the MPC object
   mpc = python.mpc.RestrictedZoneMPC(model, ntime, dt, t_step, future_steps, feet)

   #generate the reference speeds
   vref_x     = 0.1*np.ones((ntime, 1))
   vref_y     = 0.1*np.ones((ntime, 1))
   vref_theta = 0.0*np.ones((ntime, 1))
   vref       = np.hstack((vref_x, vref_y, vref_theta))

   #solutions placeholders
   CoPs          = mpc.CoP.copy()
   states        = mpc.x.copy()
   current_foots = mpc.f_current.copy()
   controls      = mpc.controls.copy()

   #main loop
   i = 0

   for t in time:
      
      results = mpc.solve(i, vref)

      states        = np.vstack((states, results[0]))
      current_foots = np.vstack((current_foots, results[1]))
      CoPs          = np.hstack((CoPs, results[2]))
      controls      = np.vstack((controls, results[3]))

      i = i + 1

   #subsample the CoM and CoP and current_foots plots - don't subsample constraints
   st, cop, tms, tm, cstr = python.plotting.subsample(feet, model, states, controls, current_foots, time_sim, dt, 0.005)

   #generate trajectories for tracking
   # For velocity set vel_accl=True
   # For acceleration set vel_accl=False
   pyx, pyy, pyz, pytheta, total_time, total_time_zero, foot_x_val, foot_y_val = python.plotting.generate_trajectories(st, current_foots, h_step, 0.005, time_sim=time_sim, vel_accl=True)

   #plots
   fig, ax = plt.subplots(1)
   plt.title('walking pattern - CoP in the restricted zone')
   ax.set_xlabel('x [m]')
   ax.set_ylabel('y [m]')
   plt.axis('equal')
   # plt.xlim(-0.2,1.2)
   # plt.xticks(np.arange(-0.2, 1.2, 0.05))
   # plt.ylim(-0.3,0.3)

   plot_foot = 1
   if (plot_foot):
      for foot in current_foots:
         foot_l.append(foot[0])
         foot_r.append(foot[1])

         #plot rotated feet
         rectangle = patches.Rectangle((foot[0]-foot_length/2, foot[1]-foot_width/2), foot_length, foot_width, color="red", fill=False)
         transform = matplotlib.transforms.Affine2D().rotate_around(foot[0], foot[1], foot[2]) + ax.transData
         rectangle.set_transform(transform)
         ax.add_patch(rectangle)

         #plot restriction zones
         circle = plt.Circle((foot[0], foot[1]), 2*mpc.zone*np.sqrt(2)/2, color='b', fill=False)
         ax.add_patch(circle)
         
         square = patches.Rectangle((foot[0] - mpc.zone, foot[1] - mpc.zone), 2*mpc.zone, 2*mpc.zone, color='y', fill=False)
         ax.add_patch(square)

      #plot CoM and CoP
      plt.plot(cop[0, :], cop[1, :], 'g')
      plt.plot(st[:, 0], st[:, 3], 'b')
      plt.plot(foot_l, foot_r, 'black', linestyle=' ', marker='+')
   
   #plot time evolution of feet trajectory coords
   plot_velo = 0
   if (plot_velo):
      fig2, ax2 = plt.subplots(1)
      plt.title('feet and CoM velocity')
      ax2.set_ylabel('velocity [m/s]')
      ax2.set_xlabel('time [s]')
      #plt.axis('equal')
      plt.xlim(0,10)
      plt.xticks(np.arange(0, 10, 0.8))
      #plt.xlim(0,5)
      #4.8s is the time needed for exactly six steps - use instead of time_sim
      plt.plot(np.linspace(0, time_sim, pyz.size), pyz.ravel(), 'r')
      plt.plot(np.linspace(0, time_sim, pytheta.size), pytheta.ravel(), 'b')

   np.savetxt('CoM_traj_x_y.txt', np.column_stack((np.arange(0, time_sim, 0.005), st[:-1, 0], st[:-1, 3])), fmt='%f', delimiter=',')
   np.savetxt('foot_traj_x_y.txt', np.column_stack((np.arange(0, time_sim, 0.005),foot_x_val.ravel(), foot_y_val.ravel())), fmt='%f', delimiter=',')
   np.savetxt('foot_l_r.txt', np.column_stack((np.arange(0, len(foot_l)),foot_l, foot_r)), fmt='%f', delimiter=',')
   np.savetxt('current_foots.txt', np.column_stack((np.arange(0, len(current_foots[::8, 0].tolist())),current_foots[::8, 0].tolist(), current_foots[::8, 1].tolist())), fmt='%f', delimiter=',')

   lx,ly, rx, ry = interpolate_from_file('current_foots.txt')
   plt.plot(lx , ly, linestyle=' ', marker='*')
   plt.plot(rx , ry, linestyle=' ', marker='*')

   #plt.plot(foot_x_val, foot_y_val, 'black', linestyle=' ', marker='o')

   # nx, ny = interpolate(0.000000,0.000000, 0.167721,0.023352)
   # plt.plot(nx , ny, linestyle=' ', marker='*')
   # nx, ny = interpolate(0.090670,-0.156900, 0.247138,-0.133547)
   # plt.plot(nx , ny, linestyle=' ', marker='*')
   # nx, ny = interpolate(0.167721,0.023352, 0.326030,0.046042)
   # plt.plot(nx , ny, linestyle=' ', marker='*')

   plt.show()

   print len(foot_l)
   print len(current_foots[::8, 0].tolist())

if __name__ == '__main__':

   main()

