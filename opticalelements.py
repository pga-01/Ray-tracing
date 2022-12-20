import scipy as sp
import numpy as np
from scipy import linalg as LA
from Display import generate_circle_by_angles


#%%
class OpticalElement:
    
    def __init__(self, 
                 z_0_1, z_0_2, 
                 curv_1, curv_2, 
                 n1, n2, 
                 # a_R, 
                 thetax, thetay, thetaz,
                 Re_1, Re_2):
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
        
        # print("thetax Sph before", self._thetax)
        
        self._thetay = thetay
        self._thetax = thetax 
        self._thetaz = thetaz

        print("thetax Sph after", self._thetax)
        print("thetay Sph after", self._thetay)
        print("thetaz Sph after", self._thetaz)

        
        self._Re_1 = Re_1
        self._Re_2 = Re_2
        
        global r1,r2,t1,t2
        self.r1 = np.sqrt(self._Re_1)
        self.r2 = np.sqrt(self._Re_2)
        self.t1 = np.sqrt(1-self.r1**2)
        self.t2 = np.sqrt(1-self.r2**2)
        
        if self.z_0_1[2] > self.z_0_2[2]:
            raise ValueError("The z-coordinate of z_0_1 must be lower than "
                             "the z-coordinate of z_0_2, i.e. spherical "
                             "surface 1 must lie closer to the x-y plane than "
                             "spherical surface 2")
            
        self._intersection = []# appended in intercept methods


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
            
    def setthetax(self, thetax):
        """
        Sets a new element tilt thetax
        """        
        self._thetax = thetax
            
    def setthetay(self, thetay):
        """
        Sets a new element tilt thetay
        """        
        self._thetay = thetay
        
    def setthetaz(self, thetaz):
        """
        Sets a new element tilt thetay
        """        
        self._thetaz = thetaz
        
    def x_rot(self, vector):
        """
        Rotates 3-D vector around x-axis
        """
        R = np.array([[1,0,0],
                      [0,np.cos(self._thetax),-np.sin(self._thetax)],
                      [0,np.sin(self._thetax),np.cos(self._thetax)]])
        
        return np.dot(R,vector)
            
    def y_rot(self, vector):
        """
        Rotates 3-D vector around y-axis
        """
        R = np.array([[np.cos(self._thetay),0,np.sin(self._thetay)],
                      [0,1,0],
                      [-np.sin(self._thetay), 0, np.cos(self._thetay)]])
        return np.dot(R,vector)

    def z_rot(self, vector):
        """
        Rotates 3-D vector around x-axis
        """
        R = np.array([[np.cos(self._thetaz),-np.sin(self._thetaz),0],
                      [np.sin(self._thetaz),np.cos(self._thetaz),0],
                      [0,0,1]])
        
        return np.dot(R,vector)
    
    def rot(self, rotpoint, thetax, thetay, thetaz):
        """
        Rotate an optical element about a given point
        """
        print("in rot")       
        self.setthetax(thetax)
        self.setthetay(thetay)
        self.setthetaz(thetaz)
        print(self._thetax)
        print(self._thetay)
        print(self._thetaz)
        centren1 = self.z_0_1-rotpoint        
        centrennew1 = rotpoint + self.z_rot(self.y_rot(self.x_rot(centren1)))
        
        centren2 = self.z_0_2-rotpoint
        centrennew2 = rotpoint + self.z_rot(self.y_rot(self.x_rot(centren2))) 
        
        self.setz_0(z_0_1 = centrennew1, z_0_2 = centrennew2)
             
    def unit_v(self, vector):
        """
        Returns a unit vector of a given vector.
        """
        
        return vector / LA.norm(vector)

    def n(self, ray, z_0):
        """
        Returns normal vector at the point of intersection of a given ray with
        the optical element's surface.
        """ 
        if self._curv_1 ==0 or self._curv_2 == 0:
            if np.all(z_0 == self.z_0_1):
                # print("thetax", self._thetax)
                # normal = self.unit_v(self.x_rot(np.array([0,0,-1])))
                # print("NORMAL Xrot", normal)
                # normal = self.unit_v(self.y_rot(np.array([0,0,-1])))
                # print("NORMAL Yrot", normal)
                # normal = self.unit_v(self.z_rot(np.array([0,0,-1])))
                # print("NORMAL Zrot", normal)
                normal = self.unit_v(self.z_rot(self.y_rot(self.x_rot(np.array([0,0,-1])))))
                # normal = self.unit_v(self.x_rot(self.y_rot([0,0,0])))
                # print("NORMAL", normal)
            elif np.all(z_0 == self.z_0_2):
                normal = self.unit_v(self.z_rot(self.y_rot(self.x_rot(np.array([0,0,1])))))
                # print("NORMAL2", normal)
        else:
            print("shouldnt")
            normal = self.intercept(ray, z_0) - self.C(z_0)
        
        return normal

    def normalz_0(self):
        """
        Returns normal vector at the point z_0_1.
        """
        normal = self.unit_v(self.z_rot(self.y_rot(self.x_rot(np.array([0,0,-1])))))
        print("normal", normal)
        return normal

    def point2plane(self, ray, z_0):
        
        centrep = (self.z_0_1 + self.z_0_2)/2 
        print("centrep", centrep)
        ponplane = centrep + LA.norm(self.z_0_1-centrep)*(self.n(ray, z_0))
        print("ponplane", ponplane)
        d = -np.dot(self.n(ray, z_0), ponplane)
        print("d", d)
        D = -(np.dot(self.n(ray, z_0), ray.p()) + d) / np.dot(self.n(ray, z_0), ray.k())
        print("D", D)
        return D

    def Snellslaw(self, ray, z_0):
        
        """
        Returns the direction of the refracted ray using Snell's law at the 
        lens's surfaces.
        """
        k1 = SphericalRefraction.unit_v(self, ray.k())# direction unit vector
        # of incidence ray
                
        n_hat = SphericalRefraction.unit_v(self, self.n(ray, z_0))#normal 
        # unit vector at surface
        if np.arccos(sp.dot(k1, n_hat))>np.pi/2:
            
            # If statement ensure n_hat has direction opposite to that of the
            # incident ray for Snell's law to apply with the right angles.
            
            n_hat = -n_hat
            
        self.theta1 = sp.arccos(sp.dot(k1, n_hat))
        # print("theta1", self.theta1*180/np.pi)
        
        if LA.norm(ray.p()-self.z_0_1) < LA.norm(ray.p()-self.z_0_2):
            
            if np.all(z_0==self.z_0_1):
            
                self.theta2 = sp.arcsin(self._n1/self._n2 * sp.sin(self.theta1)) 
                k2 = (self._n1/self._n2*k1 
                      + (self._n1/self._n2*sp.cos(self.theta1) - sp.cos(self.theta2)) 
                      * -n_hat) #from Snell's law in vector form,
            elif np.all(z_0==self.z_0_2):
                self.theta2 = sp.arcsin(self._n2/self._n1 * sp.sin(self.theta1)) 
                k2 = (self._n2/self._n1*k1 
                      + (self._n2/self._n1*sp.cos(self.theta1) - sp.cos(self.theta2)) 
                      * -n_hat) #from Snell's law in vector form,
        elif LA.norm(ray.p()-self.z_0_1) > LA.norm(ray.p()-self.z_0_2):
           
            if np.all(z_0==self.z_0_2):
            
                self.theta2 = sp.arcsin(self._n1/self._n2 * sp.sin(self.theta1)) 
                k2 = (self._n1/self._n2*k1 
                      + (self._n1/self._n2*sp.cos(self.theta1) - sp.cos(self.theta2)) 
                      * -n_hat) #from Snell's law in vector form,
            elif np.all(z_0==self.z_0_1):
                self.theta2 = sp.arcsin(self._n2/self._n1 * sp.sin(self.theta1)) 
                k2 = (self._n2/self._n1*k1 
                      + (self._n2/self._n1*sp.cos(self.theta1) - sp.cos(self.theta2)) 
                      * -n_hat) #from Snell's law in vector form,
        
        return self.unit_v(k2)    

    def reflect(self, ray, z_0):
         
        k1 = SphericalRefraction.unit_v(self, ray.k())# direction unit vector
        # of incidence ray
         
        n_hat = SphericalRefraction.unit_v(self, self.n(ray, z_0))#normal 
        print("nhat", n_hat)
        # unit vector at surface
        if np.arccos(sp.dot(k1, n_hat))>np.pi/2:
            
            # If statement ensure n_hat has direction opposite to that of the
            # incident ray for Snell's law to apply with the right angles.
            
            n_hat = -n_hat
            
        self.theta1 = sp.arccos(sp.dot(k1, n_hat))
          
        k2 = k1 + 2*np.cos(self.theta1)*(-n_hat)
         
        return self.unit_v(k2)                                                                          
             
    def propagate_ray(self, ray, action_1 = "refract", action_2 = "refract", **kwargs):
        """
        Propagates a ray through the optical element. It carries out the 
        intercept and Snell's law method first for surface 1 and then for 
        surface 2 to get positions and directions of rays accordingly.
        """
        direction = kwargs.get("direction")
        
        if not isinstance(ray, list):
            ray = [ray]
        for i in range(len(ray)):
            rayi = ray[i]
            
            if LA.norm(rayi.p()-self.z_0_1) < LA.norm(rayi.p()-self.z_0_2):
                print("up")
      
                if action_1 is not None:
                    if action_1 == "refract":
                        print("action 1 refract")
                        rayi.append(rayi.F(),
                                   self.intercept(rayi, self.z_0_1), 
                                   self.Snellslaw(rayi, self.z_0_1),
                                   rayi.A()*self.t1, 
                                   rayi.phase(),
                                   rayi.E())
                        rayi.append2(rayi.frameT(rayi.coordinates()[3][-1]-rayi.coordinates()[3][-2],
                                                 direction))  
                                 
                    elif action_1 == "reflect":
                        print("action_1 reflect")
                        rayi.append(rayi.F(),
                                   self.intercept(rayi, self.z_0_1), 
                                   self.reflect(rayi, self.z_0_1),
                                   rayi.A()*self.r1, 
                                   rayi.phase()*np.exp(1j*np.pi),
                                   rayi.E())
                        rayi.append2(rayi.frameT(rayi.coordinates()[3][-1]-rayi.coordinates()[3][-2],
                                                 direction))                                   
                    else:
                        raise ValueError("action must be either 'refract' or 'reflect'")
                
                if  action_2 is not None:
                    if action_2 == "refract":
                        print("action 2 refract")
                        rayi.append(rayi.F(),
                                   self.intercept(rayi, self.z_0_2), 
                                   self.Snellslaw(rayi, self.z_0_2),
                                   rayi.A()*self.t2, 
                                   rayi.phase(),
                                   rayi.E())
                        rayi.append2(rayi.frameT(rayi.coordinates()[3][-1]-rayi.coordinates()[3][-2],
                                                 direction))
                        
                    elif action_2 == "reflect":
                        print("action 2 reflect")
                        rayi.append(rayi.F(),
                                   self.intercept(rayi, self.z_0_2), 
                                   self.reflect(rayi, self.z_0_2),
                                   rayi.A()*self.r2, 
                                   rayi.phase()*np.exp(1j*np.pi),
                                   rayi.E())
                        rayi.append2(rayi.frameT(rayi.coordinates()[3][-1]-rayi.coordinates()[3][-2],
                                                 direction))
                                  
                    else:
                        raise ValueError("action must be either 'refract' or 'reflect'")
                        
                
            elif LA.norm(rayi.p()-self.z_0_1) > LA.norm(rayi.p()-self.z_0_2):
     
                print("down")
                
                if  action_2 is not None:
                    if action_2 == "refract":
                        print("action 2 refract")
                        rayi.append(rayi.F(),
                                    self.intercept(rayi, self.z_0_2), 
                                    self.Snellslaw(rayi, self.z_0_2),
                                    rayi.A()*self.t2, 
                                    rayi.phase(),
                                    rayi.E())
                        rayi.append2(rayi.frameT(rayi.coordinates()[3][-1]-rayi.coordinates()[3][-2],
                                                 direction))
                                   
                    elif action_2 == "reflect":
                        print("action 2 reflect")
                        rayi.append(rayi.F(),
                                    self.intercept(rayi, self.z_0_2), 
                                    self.reflect(rayi, self.z_0_2),
                                    rayi.A()*self.r2, 
                                    rayi.phase()*np.exp(1j*np.pi),
                                    rayi.E())
                        rayi.append2(rayi.frameT(rayi.coordinates()[3][-1]-rayi.coordinates()[3][-2],
                                                 direction))
                                   
                    else:
                        raise ValueError("action must be either 'refract' or 'reflect'")
                        
                if action_1 is not None:
                    if action_1 == "refract":
                        print("action 1 refract")
                        rayi.append(rayi.F(),
                                    self.intercept(ray, self.z_0_1), 
                                    self.Snellslaw(ray, self.z_0_1),
                                    rayi.A()*self.t1, 
                                    rayi.phase(),
                                    rayi.E())
                        rayi.append2(rayi.frameT(rayi.coordinates()[3][-1]-rayi.coordinates()[3][-2],
                                                 direction))
                                   
                    elif action_1 == "reflect":
                        print("action 1 reflect")
                        rayi.append(rayi.F(),
                                    self.intercept(ray, self.z_0_1), 
                                    self.reflect(ray, self.z_0_1),
                                    rayi.A()*self.r1, 
                                    rayi.phase()*np.exp(1j*np.pi),
                                    rayi.E())
                        rayi.append2(rayi.frameT(rayi.coordinates()[3][-1]-rayi.coordinates()[3][-2],
                                                 direction))
                                   
                    else:
                        raise ValueError("action must be either 'refract' or 'reflect'")
        
            return rayi._Fs, rayi._pos, rayi._directs, rayi._amps, rayi._phases     
# _____________________________________________________________________________

class SphericalRefraction(OpticalElement):
    """
    This class contains spherical optical elements and the physical processes
    that light might undergo when incident upon of these elements.

    """
    
    def __init__(self, 
                 z_0_1, z_0_2, 
                 curv_1, curv_2, 
                 n1, n2, 
                 a_R, 
                 thetay, thetax, thetaz,
                 Re_1, Re_2):
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
        
        self._a_R = a_R

                            
        if self._curv_1 !=0 and self._curv_2!=0:
            self._R_1 = abs(1/self._curv_1)
            self._R_2 = abs(1(self._curv_2))
            if self._a_R > (self._R_1 or self._R_2):
                self._a_R = min(self._R_1, self._R_2)
                # print("here")
                print(("Aperture radius, a_R, must be smaller or equal to the" + \
                      "radius of curvature of the surface with the highest curvature"
                      "The new aperture radius is %s") % (self._a_R))
        elif self._curv_1 ==0 and self._curv_2 == 0:
            print("This class deals with spherical elements, but curv_1 = {}" \
                  "and curv_2 = {}".format(self._curv1, self._curv2))
                
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
        
    
    def C(self, z_0):
        """
        Returns the centre of curvature of  a given spherical surface.
        """
        if np.all(z_0 == self.z_0_1):
            curv = self._curv_1
        elif np.all(z_0 == self.z_0_2):
            curv = self._curv_2
              
        if curv != 0:
            
             return z_0 + sp.array([0,0, 1/curv])
          
    def C_to_Ray(self, ray, z_0):
        """ 
        Returns r vector from centre of curvature to current ray position.
        """        
        r = ray.p() - self.C(z_0)
        
        return r        
       
    def intercept(self, ray, z_0):
        """
        Method calculates the first valid intercept of a ray with a spherical
        surface.
        """
        self._z_0 = z_0

        if np.all(z_0 == self.z_0_1):
             curv = self._curv_1
             R = self._R_1
        if np.all(z_0 == self.z_0_2):
             curv = self._curv_2
             R = self._R_2
             
        k_hat = SphericalRefraction.unit_v(self, ray.k())#direction unit vector
        r = self.C_to_Ray(ray, self._z_0)
                        
        if curv > 0:

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
        

    
# _____________________________________________________________________________
    
class Plane(OpticalElement):
    
    def __init__(self, 
                 z_0_1, z_0_2, 
                 n1, n2, 
                 R_1, R_2, 
                 thetax, thetay, thetaz, 
                 Re_1, Re_2):
        self._R_1 = R_1
        self._R_2 = R_2
        # print("thetax Plane before", self._thetax)
        super().__init__(z_0_1, z_0_2, 
                 0, 0, 
                 n1, n2, 
                 thetax, thetay, thetaz, 
                 Re_1, Re_2)
        print("thetax Plane after", self._thetax)
        
    def intercept(self, ray, z_0):
        """
        Method calculates the first valid intercept of a ray with a spherical
        surface.
        """
        # print("in intercept")
        self._z_0 = z_0
        
        if np.all(z_0 == self.z_0_1):
             curv = self._curv_1
             R = self._R_1
        if np.all(z_0 == self.z_0_2):
             curv = self._curv_2
             R = self._R_2
             
        k_hat = SphericalRefraction.unit_v(self, ray.k())#direction unit vector
        
        print("1st")
        lamdafactor = self.point2plane(ray, z_0)
        L = LA.norm(lamdafactor * ray.k())
        intersection= ray.p() + L*k_hat
        print("intersection", intersection)
        print("z_0", self._z_0)
        print("L", L)
        
        if LA.norm(intersection-self._z_0) > R:
            #Aperture radius ray-cut condition.
            raise ValueError(("Current ray point at position %s with "
                              "direction %s does not intercept spherical "
                              "surface with curvature %s at reference "
                              "position z_0 %s.") 
                              %(ray.p(), ray.k(), curv, self._z_0))
                
        for i in intersection:
            
            if isinstance(i, complex) == True:
                raise ValueError("Current ray point has no valid intercept")
       
        else: 
            self._intersection.append(intersection)
            
        return intersection
    
    def display(self, **kwargs):
        fig = kwargs.get("fig")
        ax = kwargs.get("ax")
        color = kwargs.get("color")
        alpha = kwargs.get("alpha")
        
        t = np.linspace(0, 2*np.pi, 100)
        C1 = self.z_0_1
        C2 = self.z_0_2
        r1 = np.linspace(0, self._R_1, 20, endpoint = False)
        r2 = np.linspace(0, self._R_2, 20, endpoint = False)
        deltar1 = r1[-1]-r1[-2]
        deltar2 = r2[-1]-r2[-2]
        print("normal", self.normalz_0())

        generate_circle_by_angles(t, C1, [self._R_1-deltar1, self._R_1], self.normalz_0(), display = True,
                                  fig = fig, ax = ax, color = "k", alpha = 1)
        generate_circle_by_angles(t, C1, r1, self.normalz_0(), display = True,
                                  fig = fig, ax = ax, color = color, alpha = alpha)
        if np.any(C1 != C2):
            print("here")
            generate_circle_by_angles(t, C2, [self._R_2-deltar2, self._R_2], self.normalz_0(), display = True,
                                  fig = fig, ax = ax, color = "k", alpha = 1)
            generate_circle_by_angles(t, C2, r2,  self.normalz_0(), display = True,
                                  fig = fig, ax = ax, color = color, alpha = alpha)
            
# _____________________________________________________________________________
        
class Mirror(Plane):
    
    def __init__(self, 
                 z_0_1,  
                 n1,
                 R_1,  
                 thetax, thetay, thetaz):
        # print("thetax Mirror before", self._thetax)

        super().__init__(z_0_1, z_0_1, 
                         n1, 1.5,
                         R_1, R_1,
                         thetax, thetay, thetaz,
                         1, 1)
        print("thetax Mirror after", self._thetax)
    
    def propagate_ray(self, ray):
        if not isinstance(ray, list):
            ray = [ray]
        for i in range(len(ray)):
            rayi = ray[i]
            
            rayi.append(rayi.F(),
                        self.intercept(rayi, self.z_0_1), 
                        self.reflect(rayi, self.z_0_1),
                        rayi.A()*self.r1, 
                        rayi.phase()*np.exp(1j*np.pi),
                        rayi.E())
        rayi.append2(rayi.frameT(rayi.coordinates()[3][-1]-rayi.coordinates()[3][-2],
                                 rayi.directions()[3][-2]))            
# _____________________________________________________________________________

class OutputPlane(Plane):
    
    """
    This class containes the "screen" on which rays can be imaged allowing
    for a means of visualizing the effects of lens refraction once propagated
    to this "screen".
    """
       
    def __init__(self, 
                 z_0_1, 
                 thetax, thetay, thetaz, 
                 shape, **kwargs):
        """
        Takes as arguments the position of the output plane in 3d space.
        kwargs: if shape = "inf", need no kwargs.
                if shape = "rec", need width W anf height H. W runs along x axis
                while H runs along y axis.
                if shape == "circle", need radius R.
        """
        
        # self._plane_pos = sp.array([plane_pos[0], plane_pos[1], plane_pos[2]])  
        self._shape = shape
        if shape == "rec":
            self._W = kwargs.get("W")
            self._H = kwargs.get("H")
        elif shape == "circle":
            self._R = kwargs.get("R")
            
        super().__init__(z_0_1, z_0_1,
                        1, 1,
                        self._R, self._R, #NEED TO GENERALIZE FOR W AND H,
                        thetax, thetay, thetaz, 
                        0, 0)   
        
        self._screen_pos = []#appended in intercept_plane method
                
    def intercept_screen(self, ray):
        """
        Returns the interception of a ray with the output plane.
        """

        print("intercept_plane")
        self._screen_point = self.intercept(ray, self.z_0_1)
        
        
        if self._shape == "inf":
            self._screen_pos.append(sp.array([self._screen_point[0], 
                                              self._screen_point[1],
                                              self._screen_point[2]]))
            return self._screen_point
            
        elif self._shape == "rec":
            if (LA.norm(self._screen_point[0]-self._plane_pos[0]) < self._W 
                and LA.norm(self._screen_point[1]-self._plane_pos[1]) < self._H):
                
                self._screen_pos.append(sp.array([self._screen_point[0], 
                                                  self._screen_point[1],
                                                  self._screen_point[2]]))
                return self._screen_point
            else:
                return None
            
        elif self._shape == "circle":
            if ((self._screen_point[0]-self.z_0_1[0])**2 + 
                (self._screen_point[1]-self.z_0_1[1])**2
                < self._R**2): 
                
                self._screen_pos.append(sp.array([self._screen_point[0], 
                                                  self._screen_point[1],
                                                  self._screen_point[2]]))
                return self._screen_point
            else:
                print("here")
                return None
        # return self._screen_point
    
    def screen_spots(self):
        """
        Extracts x, y, z and distances coordinates from screen position-arrays.
        """
        self._x_screen = sp.array([i[0] for i in self._screen_pos])
        self._y_screen = sp.array([i[1] for i in self._screen_pos])
        self._z_screen = sp.array([i[2] for i in self._screen_pos])
        
        return self._x_screen, self._y_screen, self._z_screen
        
    def propagate_ray(self, ray):
        if not isinstance(ray, list):
            ray = [ray]
        for i in range(len(ray)):
            rayi = ray[i]

            if np.all(rayi.p() == self.z_0_1) == False:
                rayi.append(rayi.F(),
                           self.intercept_screen(rayi), 
                           rayi.k(),
                           rayi.A(), 
                           rayi.phase(),
                           rayi.E())
                rayi.append2(rayi.frameT(rayi.coordinates()[3][-1]-rayi.coordinates()[3][-2],
                                         rayi.directions()[3][-2]))  
                 
                # return rayi.p()
              
            else:
                print("here2")
                #if the last z coordinate of the ray position matches that of the 
                # output plane, then stop method.
                 
                return rayi.p()

#_____________________________________________________________________________
        
class Cavity(OpticalElement):

    def __init__(self, 
                 M1, M2, 
                 L, 
                 thetax, thetay, thetaz):
        self._M1 = M1
        self._M2 = M2
        
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
        centrep = (self._M1.z_0_2 + self._M2.z_0_1)/2
        print(centrep)
        print(self._M1.z_0_1)
        print(self._M1.z_0_2)
        print(self._M2.z_0_1)
        print(self._M2.z_0_2)
        self._M1.rot(centrep, thetax = thetax, thetay = thetay, thetaz = thetaz)
        self._M2.rot(centrep, thetax = thetax, thetay = thetay, thetaz = thetaz)
        print(self._M1.z_0_1)
        print(self._M1.z_0_2)
        print(self._M2.z_0_1)
        print(self._M2.z_0_2)
        

    def resonate(self, ray, m):
        
        if isinstance(m, (int, float)):        
            m = [m]
        if not isinstance(ray, list):
            ray = [ray]
        # if len(ray) != len(m):
        #     print(("rays and m must have the same dimension but rays has shape"
        #            "{} and m has shape {}. m will be extended to shape of ray"
        #            .format(np.shape(ray), np.shape(m))))
        # if len(m)==1:
        #     m = m*len(ray)
        # counter2 = 0
        for i in range(len(ray)):
            print("HEREEEEEEEEEEE")
            rayi = ray[i]
            direction = rayi.directions()[3][-1]
            
            if LA.norm(rayi.p()-self._M1.z_0_1) < LA.norm(rayi.p()-self._M2.z_0_1):
                self._M1.propagate_ray(rayi, direction = direction)

                counter = 0
                while counter < m[i]:
                    print("HERE")                   
                    self._M2.propagate_ray(rayi, 
                                           action_1 = "reflect",
                                           action_2 = None,
                                           direction = direction)
                 
                    self._M1.propagate_ray(rayi,
                                           action_1 = None,
                                           action_2 = "reflect",
                                           direction = direction)

                    counter+=1
                self._M2.propagate_ray(rayi, direction = direction)

            elif LA.norm(rayi.p()-self._M1.z_0_1) > LA.norm(rayi.p()-self._M2.z_0_1):
                self._M2.propagate_ray(rayi, direction = direction)
              
                counter = 0
                while counter < m[i]:
                    print("HERE")
                    self._M1.propagate_ray(rayi, 
                                           action_1 = "reflect",
                                           action_2 = None,
                                           direction = direction)

                    self._M2.propagate_ray(rayi,
                                           action_1 = None,
                                           action_2 = "reflect",
                                           direction = direction)

                    counter+=1
                self._M1.propagate_ray(rayi, direction = direction)                
          
    def display(self, **kwargs):
        
        self._M1.display(**kwargs)
        self._M2.display(**kwargs)
            
        
  
            
                     
