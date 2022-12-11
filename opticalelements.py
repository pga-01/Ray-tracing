import scipy as sp
import numpy as np
from scipy import linalg as LA


#%%
class OpticalElement:
    
    def propagate_ray(self, ray):
        """
        "Propagate a ray through an optical element.
        """
        raise NotImplementedError()

   
                
            
        

        
# """
# For now, derived classes will be in the same module 
# as the base class since Python allows for this and 
# the derived classes are closely related to the base class.
# """
class SphericalRefraction(OpticalElement):
    """
    This class contains spherical optical elements and the physical processes
    that light might undergo when incident upon of these elements.

    """
    # def zero_curv(func):
    #     def wrapper(self, *args, **kwargs):
    #         returnvalue = func()
    #         if self._curv_1 == 0 or self._curv_2 == 0:
    #             return 
            
        # return wrapper
    def __init__(self, z_0_1, z_0_2, curv_1, curv_2, n1, n2, a_R, theta=0, Re_1=0, Re_2=0):
        """
        Parameters:
            
        z_0_1 and z_0_2: surface's 1 and 2 reference positions respectively
        in 3d space. They are the highest or lowest point (with respect to z 
        axis) on a spherical surface if it has negative or positive curvature 
        respectively. 
        
        curv_1 and curv_2: surface's 1 and 2 curvatures respectively. They are
        positive if centre of curvature is found at z> z_0(_1 or _2),
        negative if centre of curvature at z<z_0(_1 or _2).
        
        n1: refractive index of medium surrounding optical element.
        n2: refractive index of lens medium.
        a_R: lens aperture radius.
        R: re
        """
        self.z_0_1 = sp.array([z_0_1[0], z_0_1[1], z_0_1[2]]) 
        self.z_0_2 = sp.array([z_0_2[0], z_0_2[1], z_0_2[2]]) 
        
        self._curv_1 = curv_1 
        self._curv_2 = curv_2
        
        self._n1 = n1
        self._n2 = n2
        
        self._a_R = a_R
        
        self._theta = theta
        
        self._Re_1 = Re_1
        self._Re_2 = Re_2
        
        global r1,r2,t1,t2
        self.r1 = np.sqrt(self._Re_1)
        self.r2 = np.sqrt(self._Re_2)
        self.t1 = np.sqrt(1-self.r1**2)
        self.t2 = np.sqrt(1-self.r2**2)
        
    
        if self._curv_1 == 0:
            #This if statement deals with the special case when the curvature 
            #is zero, i.e. an infinite centre of curvature.
            self._R_1 = float("nan")
            
        else:
            self._R_1 = abs(1/self._curv_1)


        if self._curv_2 == 0:
            # This if statement deals with the special case when the curvature 
            # is zero, i.e. an infinite centre of curvature.
            
            self._R_2 = float("nan")
            
        else:
            self._R_2 = abs(1/self._curv_2)
        
        if self.z_0_1[2] > self.z_0_2[2]:
            raise ValueError("The z-coordinate of z_0_1 must be lower than "
                             "the z-coordinate of z_0_2, i.e. spherical "
                             "surface 1 must lie closer to the x-y plane than "
                             "spherical surface 2")
                            
        if self._curv_1 !=0 and  self._curv_2!=0:
            if self._a_R > (self._R_1 or self._R_2):
                self._a_R = min(self._R_1, self._R_2)
                # print("here")
                print(("Aperture radius, a_R, must be smaller or equal to the" + \
                      "radius of curvature of the surface with the highest curvature"
                      "The new aperture radius is %s") % (self._a_R))
                
        if ((self._curv_1 > 0 and self._curv_2 < 0) 
            and (self.C(z_0_1)[2] > self.C(z_0_2)[2])):
            
            # The code inside this if statement only applies for biconvex 
            # lenses.
            # It calculates the aperture radius at the intersection of the two
            # spherical surfaces and if this is smaller than the aperture 
            # radius defined in the SphericalRefraction object, then sets this
            # to the calculated aperture radius.
            
            d = abs(self.C(self.z_0_2)[2] - self.C(self.z_0_1)[2])
            calculated_a_R = (sp.sqrt(4* d*d * self._R_1*self._R_1 
                                      - (d*d - self._R_2*self._R_2 
                                         + self._R_1*self._R_1)**2) 
                              / (2 * d))
            
            if ((isinstance(calculated_a_R, complex)) == False 
                and (0 < calculated_a_R < self._a_R)):
                self._a_R = calculated_a_R

                print(("The intersection of the two spherical elements with "
                        "z_0_1: %s and z_0_2: %s formed an aperture radius "
                        "smaller than the one given. The new calculated "
                        "aperture radius is a_R: %s") 
                      % (self.z_0_1, self.z_0_2, self._a_R))
                
          
        elif (self._curv_1 > 0 and self._curv_2 == 0):
            
            # These 2 elif statements apply to a plano convex lens only.
            # They calculate the actual maximum thickness between the 
            # the spherical surface and the plane surface and if this is is 
            # smaller than the thickness defined in the SphericalRefraction
            # object, then they calculate the actual aperture radius from the 
            # actual thickness and set this as the new aperture radius.
            
            z_a_R_1 = (self._R_1 
                       - sp.sqrt(self._R_1*self._R_1 - self._a_R*self._a_R))
            thickness = abs(self.z_0_1[2] - self.z_0_2[2]) 
            if thickness < z_a_R_1:
                self._a_R = sp.sqrt(thickness * (2*self._R_1 - thickness))
                
                print(("The intersection of the two spherical elements with "
                        "z_0_1: %s and z_0_2: %s formed an aperture radius "
                        "smaller than the one given. The new calculated "
                        "aperture radius is a_R: %s") 
                      % (self.z_0_1, self.z_0_2, self._a_R))
                
        elif (self._curv_1 == 0 and self._curv_2 < 0):
            z_a_R_2 = (self._R_2 
                       - sp.sqrt(self._R_2*self._R_2 - self._a_R*self._a_R))
            thickness = abs(self.z_0_1[2] - self.z_0_2[2]) 
            if thickness < z_a_R_2:
                self._a_R = sp.sqrt(thickness * (2*self._R_2 - thickness))
                
                print(("The intersection of the two spherical elements with "
                        "z_0_1: %s and z_0_2: %s formed an aperture radius "
                        "smaller than the one given. The new calculated "
                        "aperture radius is a_R: %s") 
                      % (self.z_0_1, self.z_0_2, self._a_R))
            
        self._intersection = []# appended in intercept method
        
    def __repr__(self):
        
        return (("z_0_1: %s, z_0_2: %s, curvature_1: %s, curvature_2: %s, "
                 "n1: %s, n2: %s, aperture radius: %s")
                %(self.z_0_1, self.z_0_2, self._curv_1, self._curv_2,  
                  self._n1, self._n2, self._a_R))
        
    def __sub__(self, other):
        self._curv_1 - other._curv_1
        
        return self
        
    def __div__(self, other):
        self.curv_1 / other._curv_1
        
        return self
    
    def setz_0(self, z_0_1 = None, z_0_2 = None):
        """
        Sets a new z_0_1 and/or z_0_2 position
        """
        if z_0_1 is not None:
            self.z_0_1 = z_0_1
        if z_0_2 is not None:
            self.z_0_2 = z_0_2
            
    def settheta(self, theta):
        """
        Sets a new element tilt theta
        """        
        self._theta = theta
    def y_rot(self, vector):
        """
        Rotates 3-D vector around x-axis
        """
        R = np.array([[np.cos(self._theta),0,np.sin(self._theta)],
                      [0,1,0],
                      [-np.sin(self._theta), 0, np.cos(self._theta)]])
        return np.dot(R,vector)
            
    def unit_v(self, vector):
        """
        Returns a unit vector of a given vector.
        """
        
        return vector / LA.norm(vector)
    
    def C(self, z_0):
        """
        Returns the centre of curvature of  a given spherical surface.
        """
        self._z_0 = z_0
        
        for i in range(0, len(z_0)):
            
            if z_0[i] == self.z_0_1[i]:
                
                # This if statement links the choice of surface, z_0_1 or 
                # z_0_2, with its curvature, curv_1 and curv_2, respectively.
                
                curv = self._curv_1
                
            else:
                curv = self._curv_2
                
        if curv != 0:
            
             return z_0 + sp.array([0,0, 1/curv])
     
        else:
            
            # This else ensures that when the curvature is 0, the normal to the
            # planar optical surface is always [0, 0, -1].

            return sp.array([self._intersection[-1][0], 
                             self._intersection[-1][1], 
                             self._intersection[-1][2] - 1])
    
    def C_to_Ray(self, ray, z_0):
        """ 
        Returns r vector from centre of curvature to current ray position.
        """        
        r = ray.p() - self.C(z_0)
        
        return r        
    
    def n(self, ray, z_0):
        """
        Returns normal vector at the point of intersection of a given ray with
        the optical element's surface.
        """ 
        if self._curv_1 ==0 or self._curv_2 == 0:
            if np.all(z_0 == self.z_0_1)==True:
                normal = self.y_rot(np.array([0,0,-1]))
            elif np.all(z_0 == self.z_0_2)==True:
                normal = self.unit_v(self.y_rot(np.array([0,0,1])))
        else:
            normal = self.intercept(ray, z_0) - self.C(z_0)
        
        return normal
    
    
    def point2plane(self, ray, z_0):
        
        centrep = (self.z_0_1 + self.z_0_2)/2 
        ponplane = centrep + LA.norm(self.z_0_1-centrep)*(self.n(ray, z_0))
        d = -np.dot(self.n(ray, z_0), ponplane)
        
        # nom = abs(np.dot(self.n(ray, z_0), ray.p()) + d)
        # denom = np.sqrt(np.dot(self.n(ray, z_0), 
        #                        self.n(ray, z_0)))
        # D = nom/denom
        
        D = -(np.dot(self.n(ray, z_0), ray.p()) + d) / np.dot(self.n(ray, z_0), ray.k())
        
        return D
        
          
    def intercept(self, ray, z_0):
        """
        Method calculates the first valid intercept of a ray with a spherical
        surface.
        """
        # print("in intercept")
        self._z_0 = z_0
        
        for i in range(0, len(z_0)):
            
            if z_0[i] == self.z_0_1[i]:
                # This if statement links the choice of surface, z_0_1 or 
                # z_0_2, with its curvature, curv_1 and curv_2, respectively.
            
                curv = self._curv_1
                R = self._R_1
                
            else:              
                curv = self._curv_2
                R = self._R_2
                
        # if ray.k()[2] < 0:
        #     raise ValueError(("Current ray point at position %s with "
        #                       "direction %s does not intercept spherical "
        #                       "surface with position %s. Rays must have "
        #                       "direction with a positive z-component and must "
        #                       "have an initial position outside the lens.")
        #                      %(ray.p(), ray.k(), self._z_0))

        k_hat = SphericalRefraction.unit_v(self, ray.k())#direction unit vector
        
        if curv == 0: #this means the surface lies on a plane with z = z_0
            
            # Calculates the length (in coherence with the methods used for
            # other curvatures) to a plane by finding an appropiate constant 
            # lambda that multiplies the direction vector of the ray towards 
            # the plane z= z_0.
            
            # lamdafactor = (self._z_0[2] - ray.p()[2]) / ray.k()[2]
            
            lamdafactor = self.point2plane(ray, z_0)
            # print("Lamdafactor", lamdafactor)
            #Pythagorean theorem at z_a_R
            entry_radius_z_0 = ((ray.p()[0] + lamdafactor*ray.k()[0])**2 
                                + (ray.p()[1] + lamdafactor*ray.k()[1])**2)
            if  entry_radius_z_0 > self._a_R * self._a_R:
                #Aperture radius ray-cut condition.
                raise ValueError(("Current ray point at position %s with "
                                  "direction %s does not intercept spherical "
                                  "surface with curvature %s at reference "
                                  "position z_0 %s.") 
                                 %(ray.p(), ray.k(), curv, self._z_0))
        
            L = LA.norm(lamdafactor * ray.k())
            
        else: 
            r = self.C_to_Ray(ray, self._z_0)
                        
        if curv > 0.:

            z_a_R = self._z_0[2] + (R - sp.sqrt(R*R - self._a_R*self._a_R))# z
            # coordinate where aperture radius of lens is achieved
            lamda_z_a_R = (z_a_R - ray.p()[2]) / ray.k()[2]#scaling factor to
            # z_a_R
            #Pythagorean theorem at z_a_R
            entry_radius_z_a_R = ((ray.p()[0] + lamda_z_a_R*ray.k()[0])**2 
                                  + (ray.p()[1] + lamda_z_a_R*ray.k()[1])**2)
                            
            if entry_radius_z_a_R > self._a_R * self._a_R:
                raise ValueError(("Current ray point at position %s with "
                                  "direction %s does not intercept spherical "
                                  "surface with curvature %s at reference "
                                  "position z_0 %s.") 
                                 %(ray.p(), ray.k(), curv, self._z_0))
               
            L = (-sp.dot(r, k_hat) - 
                 sp.sqrt(sp.dot(r, k_hat)**2 - (LA.norm(r)**2 - R*R)))
            
        elif curv < 0.:
            
            z_a_R = self._z_0[2] - (R - sp.sqrt(R*R - self._a_R*self._a_R))#z 
            # coordinate at which the aperture radius is achieved
            lamda_z_a_R = (z_a_R - ray.p()[2]) / ray.k()[2]#scaling factor to
            # z_a_R
            #Pythagorean theorem at z_a_R
            entry_radius_z_a_R = ((ray.p()[0] + lamda_z_a_R*ray.k()[0])**2 
                                  + (ray.p()[1] + lamda_z_a_R*ray.k()[1])**2)
                
            if entry_radius_z_a_R > self._a_R * self._a_R:
                raise ValueError(("Current ray point at position %s with "
                                  "direction %s does not intercept spherical "
                                  "surface with curvature %s at reference "
                                  "position z_0 %s.") 
                                 %(ray.p(), ray.k(), curv, self._z_0))
                
            L = (-sp.dot(r, k_hat) + 
                 sp.sqrt(sp.dot(r, k_hat)**2 - (LA.norm(r)**2 - R*R)))
                  
        intersection = ray.p() + L*k_hat
        
        for i in intersection:
            
            if isinstance(i, complex) == True:
                raise ValueError("Current ray point has no valid intercept")
       
        else: 
            self._intersection.append(intersection)
            
        return intersection
        

    def Snellslaw(self, ray, z_0):
        
        """
        Returns the direction of the refracted ray using Snell's law at the 
        lens's surfaces.
        """
        k1 = SphericalRefraction.unit_v(self, ray.k())# direction unit vector
        # of incidence ray
        
        # print("k1", k1)
        
        n_hat = SphericalRefraction.unit_v(self, self.n(ray, z_0))#normal 
        # unit vector at surface
        if sp.sign(n_hat[2]) != sp.sign(ray.k()[2]):
            
            # If statement ensure n_hat has direction opposite to that of the
            # incident ray for Snell's law to apply with the right angles.
            
            n_hat = -n_hat
        # print("nhat", n_hat)
            
        self.theta1 = sp.arccos(sp.dot(k1, n_hat))
        # print("theta1", self.theta1*180/np.pi)
        
        if ray.k()[2] > 0:
            
            if np.all(z_0==self.z_0_1):
            
                self.theta2 = sp.arcsin(self._n1/self._n2 * sp.sin(self.theta1)) 
                # print("kz>0, z_0_1, theta2", self.theta2*180/np.pi)
                k2 = (self._n1/self._n2*k1 
                      + (self._n1/self._n2*sp.cos(self.theta1) - sp.cos(self.theta2)) 
                      * -n_hat) #from Snell's law in vector form,
                # print(np.arccos(np.dot(self.unit_v(k2), n_hat))*180/np.pi)
            elif np.all(z_0==self.z_0_2):
                # print("kz>0, z_0_2, theta1", self.theta1*180/np.pi)
                self.theta2 = sp.arcsin(self._n2/self._n1 * sp.sin(self.theta1)) 
                # print("kz>0, z_0_2, theta2", self.theta2*180/np.pi)
                k2 = (self._n2/self._n1*k1 
                      + (self._n2/self._n1*sp.cos(self.theta1) - sp.cos(self.theta2)) 
                      * -n_hat) #from Snell's law in vector form,
        elif ray.k()[2] < 0:
            # print("down")
           
            if np.all(z_0==self.z_0_2):
            
                self.theta2 = sp.arcsin(self._n1/self._n2 * sp.sin(self.theta1)) 
                # print("kz>0, z_0_1, theta2", self.theta2*180/np.pi)
                k2 = (self._n1/self._n2*k1 
                      + (self._n1/self._n2*sp.cos(self.theta1) - sp.cos(self.theta2)) 
                      * -n_hat) #from Snell's law in vector form,
                # print(np.arccos(np.dot(self.unit_v(k2), n_hat))*180/np.pi)
            elif np.all(z_0==self.z_0_1):
                # print("kz>0, z_0_2, theta1", self.theta1*180/np.pi)
                self.theta2 = sp.arcsin(self._n2/self._n1 * sp.sin(self.theta1)) 
                # print("kz>0, z_0_2, theta2", self.theta2*180/np.pi)
                k2 = (self._n2/self._n1*k1 
                      + (self._n2/self._n1*sp.cos(self.theta1) - sp.cos(self.theta2)) 
                      * -n_hat) #from Snell's law in vector form,
        # print("normk2", LA.norm(self.unit_v(k2)))
        # print("k2", self.unit_v(k2))
        
        return self.unit_v(k2)
                      
        
        
            
            
            

        
             
         
        
         # if (self._curv_1 < 0  
         #     or (self._curv_1 == 0 and self._curv_2 > 0) 
         #     or (sp.sign(self._curv_1) == sp.sign(self._curv_2))):
             
         #     #This if statement considers all types of diverging lenses.
             
         #     if (len(self._intersection) > 2 
         #         and (self._intersection[-3][2] 
         #              <= ray.p()[2] 
         #              <= self._intersection[-1][2])):
                 
         #         # Only diverging lenses considered here.
         #         # This if statement serves for attributing correctly the 
         #         # refractive indices of the mediums the ray encounter 
         #         # thrughout its propagation. e.g an incident ray into a 
         #         # sphereical element from n1 to n2 will then exit this element
         #         # from n2 to n1. 
                 
         #         (self._n1, self._n2) = (self._n2, self._n1)
                 
         #         self.theta2 = (sp.arcsin(self._n1 * sp.sin(self.theta1) 
         #                                  / self._n2))    
         #         print("theta2 here1", self.theta2*180/np.pi)

                    
         #         k2 = (self._n1/self._n2 * k1 
         #               + (self._n1/self._n2 * sp.cos(self.theta1) 
         #                  - sp.cos(self.theta2)) 
         #               * n_hat) # From Snell's law in vector form
                              
         #         (self._n1, self._n2) = (self._n2, self._n1)# refractive index 
         #         # swap again ready for next medium.
                                                                
         #     else:
         #         self.theta2 = (sp.arcsin(self._n1 * sp.sin(self.theta1) 
         #                                  / self._n2))
         #         print("theta2 here2", self.theta2*180/np.pi)                        
         #         k2 = (self._n1/self._n2 * k1 
         #               + (self._n1/self._n2 * sp.cos(self.theta1) 
         #                  - sp.cos(self.theta2)) 
         #               * n_hat) 
             
         # elif self.z_0_1[2] <= ray.p()[2] <= self.z_0_2[2]: 
             
         #     # Only converging lenses considered here.
         #     # Same purpose as in if statement nested in previous if statement.
            
         #     (self._n1, self._n2) = (self._n2, self._n1)
             
         #     self.theta2 = (sp.arcsin(self._n1 * sp.sin(self.theta1) 
         #                              / self._n2))
         #     print(self.theta2*180/np.pi)
         #     k2 = (self._n1/self._n2 * k1 
         #           + (self._n1/self._n2 * sp.cos(self.theta1) 
         #              - sp.cos(self.theta2)) 
         #           * n_hat) #from Snell's law in vector form,
             
         #     (self._n1, self._n2) = (self._n2, self._n1)# refractive index swap
         #     # again ready for next medium.
             
         # else:
         #     self.theta2 = sp.arcsin(self._n1 * sp.sin(self.theta1) / self._n2)
         #     k2 = (self._n1/self._n2 * k1 
         #           + (self._n1/self._n2 * sp.cos(self.theta1) 
         #              - sp.cos(self.theta2)) 
         #           * n_hat) #from Snell's law in vector form
             
         # if sp.sin(self.theta1) > self._n2 / self._n1:
         #     k2 = (k1 - 2*sp.dot(n_hat, k1) * n_hat)
         #     raise ValueError("Total internal reflection taking place.")
             
         #     return SphericalRefraction.unit_v(self,k2)
    
         # return SphericalRefraction.unit_v(self,k2)
     
    def reflect(self, ray, z_0):
        
        
        # print(ray.p())
        
        k1 = SphericalRefraction.unit_v(self, ray.k())# direction unit vector
        # of incidence ray
         
        n_hat = SphericalRefraction.unit_v(self, self.n(ray, z_0))#normal 
        # unit vector at surface
        if sp.sign(n_hat[2]) != sp.sign(ray.k()[2]):
            
            # If statement ensure n_hat has direction opposite to that of the
            # incident ray for Snell's law to apply with the right angles.
            
            n_hat = -n_hat
        # print("nhat", n_hat)
            
        self.theta1 = sp.arccos(sp.dot(k1, n_hat))
        # print("theta1", self.theta1*180/np.pi)
          
        k2 = k1 + 2*np.cos(self.theta1)*(-n_hat)
        # print("normk2", LA.norm(self.unit_v(k2)))
        # print("k2", self.unit_v(k2))
         
          # p0 = ray.p()
          # lamda = LA.norm(self.intercept(ray, z_0) - p0)
          # r0 = p0 - 2*lamda*np.cos(self.theta1)*n_hat
          # k2 = SphericalRefraction.unit_v(self, self.intercept(ray, z_0)-r0)
         
        return self.unit_v(k2)                                                                          
             
     
    def propagate_ray(self, ray, action_1 = "refract", action_2 = "refract"):
        """
        Propagates a ray through the optical element. It carries out the 
        intercept and Snell's law method first for surface 1 and then for 
        surface 2 to get positions and directions of rays accordingly.
        """
        if ray.k()[2] >0:
            # print("up")
            # pass
        # elif ray.k()[2] < 0:
        #     print("back")
        #     (action_1, action_2 == action_2, action_1)
        #     (self.z_0_1, self.z_0_2 == self.z_0_2, self.z_0_1)
        #     (self.t1, self.t2 == self.t2, self.t1)
        #     (self.r1, self.r2 == self.r2, self.r1)
        #     print("action1", action_1)
        #     print("action2", action_2)
            
            if action_1 is not None:
                if action_1 == "refract":
                    # print("action 1 refract")
                    ray.append(ray.F(),
                               self.intercept(ray, self.z_0_1), 
                               self.Snellslaw(ray, self.z_0_1),
                               ray.A()*self.t1, 
                               ray.phase(),
                               ray.E())
                             
                elif action_1 == "reflect":
                    # print("action_1 reflect")
                    ray.append(ray.F(),
                               self.intercept(ray, self.z_0_1), 
                               self.reflect(ray, self.z_0_1),
                               ray.A()*self.r1, 
                               ray.phase()*np.exp(1j*np.pi),
                               ray.E())
                               
                else:
                    raise ValueError("action must be either 'refract' or 'reflect'")
            
            if  action_2 is not None:
                if action_2 == "refract":
                    # print("action 2 refract")
                    ray.append(ray.F(),
                               self.intercept(ray, self.z_0_2), 
                               self.Snellslaw(ray, self.z_0_2),
                               ray.A()*self.t2, 
                               ray.phase(),
                               ray.E())
                    
                elif action_2 == "reflect":
                    # print("action 2 reflect")
                    ray.append(ray.F(),
                               self.intercept(ray, self.z_0_2), 
                               self.reflect(ray, self.z_0_2),
                               ray.A()*self.r2, 
                               ray.phase()*np.exp(1j*np.pi),
                               ray.E())
                              
                else:
                    raise ValueError("action must be either 'refract' or 'reflect'")
                    
            
        elif ray.k()[2] <0:
            # print("down")
            
            if  action_2 is not None:
                if action_2 == "refract":
                    ray.append(ray.F(),
                                self.intercept(ray, self.z_0_2), 
                                self.Snellslaw(ray, self.z_0_2),
                                ray.A()*self.t2, 
                                ray.phase(),
                                ray.E())
                               
                elif action_2 == "reflect":
                    ray.append(ray.F(),
                                self.intercept(ray, self.z_0_2), 
                                self.reflect(ray, self.z_0_2),
                                ray.A()*self.r2, 
                                ray.phase()*np.exp(1j*np.pi),
                                ray.E())
                               
                else:
                    raise ValueError("action must be either 'refract' or 'reflect'")
                    
            if action_1 is not None:
                if action_1 == "refract":
                    ray.append(ray.F(),
                                self.intercept(ray, self.z_0_1), 
                                self.Snellslaw(ray, self.z_0_1),
                                ray.A()*self.t1, 
                                ray.phase(),
                                ray.E())
                               
                elif action_1 == "reflect":
                    ray.append(ray.F(),
                                self.intercept(ray, self.z_0_1), 
                                self.reflect(ray, self.z_0_1),
                                ray.A()*self.r1, 
                                ray.phase()*np.exp(1j*np.pi),
                                ray.E())
                               
                else:
                    raise ValueError("action must be either 'refract' or 'reflect'")
            
        
        return ray._Fs, ray._pos, ray._directs, ray._amps, ray._phases

    
class OutputPlane(SphericalRefraction):
    
    """
    This class containes the "screen" on which rays can be imaged allowing
    for a means of visualizing the effects of lens refraction once propagated
    to this "screen".
    """
       
    def __init__(self, plane_pos, shape="inf", **kwargs):
        
        """
        Takes as arguments the position of the output plane in 3d space.
        kwargs: if shape = "inf", need no kwargs.
                if shape = "rec", need width W anf height H. W runs along x axis
                while H runs along y axis.
                if shape == "circle", need radius R.
        """
        self._plane_pos = sp.array([plane_pos[0], plane_pos[1], plane_pos[2]])  
        self._shape = shape
        if shape == "rec":
            self._W = kwargs.get("W")
            self._H = kwargs.get("H")
        elif shape == "circle":
            self._R = kwargs.get("R")
            
        self._screen_pos = []#appended in intercept_plane method
                
        

    def intercept_plane(self, ray):
        """
        Returns the interception of a ray with the output plane.
        """
        normal_plane = sp.array([0,0,-1])
        lamda_plane = (float(sp.dot((self._plane_pos - ray.p()), normal_plane) 
                             / sp.dot(ray.k(), normal_plane)))#scaling factor
        # to plane.
        self._screen_point = ray.p() + ray.k()*lamda_plane
        
        if self._shape == "inf":
            self._screen_pos.append(sp.array([self._screen_point[0], 
                                          self._screen_point[1],
                                          self._screen_point[2]]))
            return self._screen_point
            
        elif self._shape == "rec":
            if (LA.norm(self._screen_point[0]-self._plane_pos[0]) < self._W 
                and LA.norm(self._screen_point[1]-self._plane_pos[0]) < self._H):
                
                self._screen_pos.append(sp.array([self._screen_point[0], 
                                          self._screen_point[1],
                                          self._screen_point[2]]))
                return self._screen_point
            else:
                return None
        elif self._shape == "circle":
            if ((self._screen_point[0]-self._plane_pos[0])**2 + 
                (self._screen_point[1]-self._plane_pos[0])**2
                < self._R**2): 
                
                self._screen_pos.append(sp.array([self._screen_point[0], 
                                          self._screen_point[1],
                                          self._screen_point[2]]))
                return self._screen_point
            else:
                return None
        # return self._screen_point
    
    def screen_spots(self):
        """
        Extracts x,y and z coordinates from screen position-arrays.
        """
        self._x_screen = sp.array([i[0] for i in self._screen_pos])
        self._y_screen = sp.array([i[1] for i in self._screen_pos])
        self._z_screen = sp.array([i[2] for i in self._screen_pos])
        
        return self._x_screen, self._y_screen, self._z_screen
        

    def propagate_ray(self, ray):
        
        if ray.p()[2] != self._plane_pos[2]:
             ray.append(ray.F(),
                        self.intercept_plane(ray), 
                        ray.k(),
                        ray.A(), 
                        ray.phase(),
                        ray.E())
             
             return ray.p()
              
        else:
            #if the last z coordinate of the ray position matches that of the 
            # output plane, then stop method.
             
            return ray.p()
        
class Cavity(SphericalRefraction):
    
    #No need to define an __init__() because it's same as for 
    #superclass.
    
    def __init__(self, M1, M2, outputplane, L=None, theta=0):
        
        self._M1 = M1
        self._M2 = M2
        self._outputplane = outputplane
        # print(self._M1.z_0_1, self._M1.z_0_2)
        # print(self._M2.z_0_1, self._M2.z_0_2)
        # print(self._M1._theta, self._M1._theta)
        # print(self._M2._theta, self._M2._theta)
        
        if L is not None:
            if type(L)==float or int:
                deltaL =  (self._M1.z_0_2 + np.array([0,0,L]) 
                           - self._M2.z_0_1)
                self._M2.setz_0(z_0_1 = self._M2.z_0_1+deltaL, 
                                   z_0_2 = self._M2.z_0_2+deltaL)
            elif len(L) != 0:
                deltaL =  (self._M1.z_0_2 + np.array([0,0,L[0]]) 
                           - self._M2.z_0_1)
                self._M2.setz_0(z_0_1 = self._M2.z_0_1+deltaL, 
                                   z_0_2 = self._M2.z_0_2+deltaL)
        self._M1.settheta(theta)
        self._M2.settheta(theta)
        
    def resonate(self, rays):
            
        
        for i in range(len(rays)):
            print("i", i)
            # print(self._M1.z_0_1, self._M1.z_0_2)
            # print(self._M2.z_0_1, self._M2.z_0_2)
            # print(self._M1._theta, self._M1._theta)
            # print(self._M2._theta, self._M2._theta)  
            rayi = rays[i]
            self._M1.propagate_ray(rayi)
            counter = 0
            while counter < i:
                # print("i",i)
                # print("self.M2", i)
                self._M2.propagate_ray(rayi, 
                                       action_1 = "reflect",
                                       action_2 = None)
                # print("self.M1", i)
                self._M1.propagate_ray(rayi,
                                       action_1 = None,
                                       action_2 = "reflect")
                counter+=1
            self._M2.propagate_ray(rayi)
            self._outputplane.propagate_ray(rayi)
        
    
    # def 
                                                                         

    
    # def setL(self, comp1, comp2, L):
    #     if L is not None:
    #         if type(L)==float:
    #                 self._comp2.z_0_1 = self._comp1.z_0_2 + L
    #         elif len(L) != 0:
    #                 print("here")
    #                 self._comp2.z_0_1 = self._comp1.z_0_2 + L[0]
        
    
        

            
    
    # def L(self):
    #     self._comp2.z_0_1 = self._comp1.z_0_2 + L
    
        

        
            
                     
