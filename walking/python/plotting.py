#auxilliary routines

import numpy as np

def apply2DRotation(pointx, pointy, theta):
   """apply the 2D rotation matrix to a point
   
   """

   if isinstance(pointx, list) and isinstance(pointy, list):
      #if we pass a list of points and one angle
      r_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
      for idx, item in enumerate(zip(pointx, pointy)):
         point = np.array([[pointx[idx], pointy[idx]]]).T
         point = np.dot(r_mat, point)
         pointx[idx], pointy[idx] = point[0][0], point[1][0]
      return [pointx, pointy]
   elif isinstance(pointx, np.ndarray) and isinstance(pointy, np.ndarray) and isinstance(theta, np.ndarray):
      #if we pass a list of points and angles
      for idx, item in enumerate(zip(pointx, pointy, theta)):
         r_mat = np.array([[np.cos(theta[idx]), -np.sin(theta[idx])], [np.sin(theta[idx]), np.cos(theta[idx])]])
         point = np.array([[pointx[idx], pointy[idx]]]).T
         point = np.dot(r_mat, point)
         pointx[idx], pointy[idx] = point[0][0], point[1][0]
      return [pointx, pointy]
   else:
      #if we pass sth else
      raise Exception("invalid input")

def subsample(feet, model, states, controls, current_foots, time_sim, olddt, newdt):
   """subsample the data between given points
      using the model matrices also produce CoP constraints for plotting

   """

   #placeholders for the results
   st  = states[0, np.newaxis]
   cop = np.dot(model.D(newdt), st.T)

   for state, control in zip(states, controls[1:, :]):
      s = state[np.newaxis].T
      #subsample
      for i in xrange(0, int(olddt/newdt)):
         s   = np.dot(model.A(newdt), s) + np.dot(model.B(newdt), control[np.newaxis].T)
         st  = np.vstack((st, s.T))
         cop = np.hstack((cop, np.dot(model.D(newdt), s)))

   #new subsampled time vector
   tms = np.linspace(0, time_sim, cop.shape[1])

   #not subsampled time vector for plotting CoP constraints
   tm  = np.linspace(0, time_sim, current_foots.shape[0])

   #compute the restrained CoP zone
   zone        = (np.min(feet)/np.sqrt(2.))/2.
   CoP_ubounds = np.array([ zone,  zone])
   CoP_lbounds = np.array([-zone, -zone])

   #[x+up, y+up, x+down, y+down] - we don't subsample constraints
   constraints       = np.hstack((current_foots, current_foots))
   constraints[:, 0] = current_foots[:, 0] + CoP_ubounds[0]
   constraints[:, 1] = current_foots[:, 1] + CoP_ubounds[1]
   constraints[:, 2] = current_foots[:, 0] + CoP_lbounds[0]
   constraints[:, 3] = current_foots[:, 1] + CoP_lbounds[1]

   #return subsampled CoPs and states
   return [st, cop, tms, tm, constraints]

def generate_trajectories(state, current_foots, h_step, dt, time_sim, save=True, vel_accl=False):
   """      
      generate foot trajectories (x, y, z, theta - doubledots)
      for tracking with whole body controller
      output -> accelerations

   """

   #support states durations
   ss  = 0.7
   tds = 0.1

   #collect values from feet coords
   x     = current_foots[::8, 0]
   y     = current_foots[::8, 1]
   theta = current_foots[::8, 2]
   
   #build time vector for x, y, z, theta
   time_nzero = np.linspace(0, ss, ss/dt)
   time_zero  = np.linspace(0, tds, tds/dt)
   time_z     = np.linspace(0, ss/2, (ss/2)/dt)
   pzero      = np.poly1d([0])

   use_vel = vel_accl
   if(use_vel == True):
      #first step handling
      #build polynomials
      pxdd_temp  = np.poly1d([(-12.0/(ss**3))*(x[1] - x[0]), (6.0/(ss**2))*(x[1] - x[0])])
      pxdd = np.polyint(pxdd_temp)
      #y case
      pydd_temp  = np.poly1d([(-12.0/(ss**3))*(y[1] - y[1]), (6.0/(ss**2))*(y[1] - y[1])])
      pydd = np.polyint(pydd_temp)
      #theta case
      ptdd_temp  = np.poly1d([(-12.0/(ss**3))*(theta[1] - theta[0]), (6.0/(ss**2))*(theta[1] - theta[0])])
      ptdd = np.polyint(ptdd_temp)
      #z case
      pzdd1_temp = np.poly1d([(-12.0/((ss/2)**3))*(h_step - 0.0), (6.0/((ss/2)**2))*(h_step - 0.0)])
      pzdd2_temp = np.poly1d([(-12.0/((ss/2)**3))*(0.0 - h_step), (6.0/((ss/2)**2))*(0.0 - h_step)])
      pzdd1 = np.polyint(pzdd1_temp)
      pzdd2 = np.polyint(pzdd2_temp)

      #evaluate polynomials
      pyx     = np.hstack((pxdd(time_nzero), pzero(time_zero))) 
      pyy     = np.hstack((pydd(time_nzero), pzero(time_zero)))
      pytheta = np.hstack((ptdd(time_nzero), pzero(time_zero)))
      pyz     = np.hstack((pzdd1(time_z), pzdd2(time_z), pzero(time_zero)))
      
      for idx in xrange(x.shape[0]-2):
         #build polynomials
         pxdd_temp    = np.polyint(np.poly1d([(-12.0/(ss**3))*(x[idx+2] - x[idx]), (6.0/(ss**2))*(x[idx+2] - x[idx])]))
         #y case
         pydd_temp    = np.polyint(np.poly1d([(-12.0/(ss**3))*(y[idx+2] - y[idx]), (6.0/(ss**2))*(y[idx+2] - y[idx])]))
         #theta case
         ptdd_temp    = np.polyint(np.poly1d([(-12.0/(ss**3))*(theta[idx+2] - theta[idx]), (6.0/(ss**2))*(theta[idx+2] - theta[idx])]))
         #z case
         pzdd1_temp   = np.polyint(np.poly1d([(-12.0/((ss/2)**3))*(h_step - 0.0), (6.0/((ss/2)**2))*(h_step - 0.0)]))
         pzdd2_temp   = np.polyint(np.poly1d([(-12.0/((ss/2)**3))*(0.0 - h_step), (6.0/((ss/2)**2))*(0.0 - h_step)]))
         #evaluate polynomials
         pyx     = np.vstack((pyx, np.hstack((pxdd_temp(time_nzero), pzero(time_zero))))) 
         pyy     = np.vstack((pyy, np.hstack((pydd_temp(time_nzero), pzero(time_zero)))))
         pytheta = np.vstack((pytheta, np.hstack((ptdd_temp(time_nzero), pzero(time_zero)))))
         pyz     = np.vstack((pyz, np.hstack((pzdd1_temp(time_z), pzdd2_temp(time_z), pzero(time_zero)))))

   if(use_vel == False):
      #first step handling
      #build polynomials
      pxdd  = np.poly1d([(-12.0/(ss**3))*(x[1] - x[0]), (6.0/(ss**2))*(x[1] - x[0])])
      #y case
      pydd  = np.poly1d([(-12.0/(ss**3))*(y[1] - y[1]), (6.0/(ss**2))*(y[1] - y[1])])
      #theta case
      ptdd  = np.poly1d([(-12.0/(ss**3))*(theta[1] - theta[0]), (6.0/(ss**2))*(theta[1] - theta[0])])
      #z case
      pzdd1 = np.poly1d([(-12.0/((ss/2)**3))*(h_step - 0.0), (6.0/((ss/2)**2))*(h_step - 0.0)])
      pzdd2 = np.poly1d([(-12.0/((ss/2)**3))*(0.0 - h_step), (6.0/((ss/2)**2))*(0.0 - h_step)])

      #evaluate polynomials
      pyx     = np.hstack((pxdd(time_nzero), pzero(time_zero))) 
      pyy     = np.hstack((pydd(time_nzero), pzero(time_zero)))
      pytheta = np.hstack((ptdd(time_nzero), pzero(time_zero)))
      pyz     = np.hstack((pzdd1(time_z), pzdd2(time_z), pzero(time_zero)))
      
      for idx in xrange(x.shape[0]-2):
         #build polynomials
         pxdd    = np.poly1d([(-12.0/(ss**3))*(x[idx+2] - x[idx]), (6.0/(ss**2))*(x[idx+2] - x[idx])])
         #y case
         pydd    = np.poly1d([(-12.0/(ss**3))*(y[idx+2] - y[idx]), (6.0/(ss**2))*(y[idx+2] - y[idx])])
         #theta case
         ptdd    = np.poly1d([(-12.0/(ss**3))*(theta[idx+2] - theta[idx]), (6.0/(ss**2))*(theta[idx+2] - theta[idx])])
         #z case
         pzdd1   = np.poly1d([(-12.0/((ss/2)**3))*(h_step - 0.0), (6.0/((ss/2)**2))*(h_step - 0.0)])
         pzdd2   = np.poly1d([(-12.0/((ss/2)**3))*(0.0 - h_step), (6.0/((ss/2)**2))*(0.0 - h_step)])
         #evaluate polynomials
         pyx     = np.vstack((pyx, np.hstack((pxdd(time_nzero), pzero(time_zero))))) 
         pyy     = np.vstack((pyy, np.hstack((pydd(time_nzero), pzero(time_zero)))))
         pytheta = np.vstack((pytheta, np.hstack((ptdd(time_nzero), pzero(time_zero)))))
         pyz     = np.vstack((pyz, np.hstack((pzdd1(time_z), pzdd2(time_z), pzero(time_zero)))))


   # pyx = np.around(pyx,5)
   # pyy = np.around(pyy,5)
   # pyz = np.around(pyz,5)
   # pytheta = np.around(pytheta,5)

   if save:
      #save stuff for whole body motion
      np.savetxt('fx.txt', pyx.ravel(), fmt='%f', delimiter=',')
      np.savetxt('fy.txt', pyy.ravel(), fmt='%f', delimiter=',')
      np.savetxt('fz.txt', pyz.ravel(), fmt='%f', delimiter=',')
      np.savetxt('ftheta.txt', pytheta.ravel(), fmt='%f', delimiter=',')   
      np.savetxt('xcom.txt', state[:pyx.ravel().shape[0], 2], fmt='%f', delimiter=',')
      np.savetxt('ycom.txt', state[:pyx.ravel().shape[0], 5], fmt='%f', delimiter=',')
      np.savetxt('thetacom.txt', state[:pyx.ravel().shape[0], 8], fmt='%f', delimiter=',')
      np.savetxt('foot_vel_x_y.txt', np.column_stack((np.arange(0, time_sim, 0.005), np.around(pyx.ravel(),5), np.around(pyy.ravel(),5))), fmt='%f', delimiter=',')
      np.savetxt('CoM_vel_x_y.txt', np.column_stack((np.arange(0, time_sim, 0.005), np.around(state[:pyx.ravel().shape[0], 2],5), np.around(state[:pyx.ravel().shape[0], 5],5))), fmt='%f', delimiter=',')

   return [pyx, pyy, pyz, pytheta, time_nzero, time_zero]

