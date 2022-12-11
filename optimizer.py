from ray import Ray
from opticalelements import (OpticalElement, SphericalRefraction)
                                   
import scipy as sp
import scipy.optimize as op
#%%

class Optimization(OpticalElement):
    """
    Class contains an optimization method for the curvature of a lens and
    methods of data analysis such as the RMS or GEO radius.
    """       

    def __init__(self):
        
        # self._all_optimized_curvs = []
        self._optimized_curvatures = []
        
    def optimize(self, bundle, bundle_direct, lens, screen):
        """
        Method carries out an optimziation of the lens' curvatures by 
        minimising the root mean square radius that the rays form with the 
        centroid ray at the output screen. 
        It has as arguments a Bundle object, its direction vector,
        a SphericalRefraction object and an OutputPlane object which should 
        all be previously created. 
        Method will use the parameters from the SphericalRefraction object 
        except for the curvatures which it modifies. 
        Method returns the root mean square 
        radius.
        
        """
        bundle_direct = sp.array([bundle_direct[0], 
                                  bundle_direct[1], 
                                  bundle_direct[2]])
        
        initial_curvatures = [lens.z_0_1, lens.z_0_2]

        def optimizing_func(curvatures):
                                   
            newlens = SphericalRefraction(lens.z_0_1, lens.z_0_2, 
                                          curvatures[0], curvatures[1], 
                                          lens._n1, lens._n2, 
                                          lens._a_R)
            
            MS = 0
            for ray_i in bundle.spots():
                centroid = Ray(bundle.spots()[0], bundle_direct)
                beam = Ray(ray_i, bundle_direct)
                
                newlens.propagate_ray(beam)
                newlens.propagate_ray(centroid)
                
                screen.propagate_ray(beam)
                screen.propagate_ray(centroid)
                
                mean_squared = (((beam.p()[0]-centroid.p()[0]) 
                                 * (beam.p()[0]-centroid.p()[0]) 
                                +                                 
                                (beam.p()[1]-centroid.p()[1]) 
                                * (beam.p()[1]-centroid.p()[1])) 
                                
                                / bundle._n)
                
                MS += mean_squared
                
            RMS = sp.sqrt(MS)
            self._optimized_curvs = sp.array([curvatures[0], curvatures[1]])
            
            
            return RMS
        
       
        op.fmin(optimizing_func, initial_curvatures, disp = True, 
                full_output = True) 
        # print("The optimized curvature of the first lens surface is "
        #            "curv_1: %s and for the other lens surface is: %s" 
        #            % (self._optimized_curvs[0], self._optimized_curvs[1]))
        self._optimized_curvatures.append(self._optimized_curvs)
        
    def __repr__(self):
        
        return ("%s: %s, %s: %s" 
                %("optimised curv_1", self._optimized_curvs[0], 
                  "optimised curv_2", self._optimized_curvs[1]))
    
    def __str__(self):
        
        return "(%s, %s)"%(self._optimized_curvs[0], self._optimized_curvs[1])
    
    def opt_curvatures(self):
        """
        Method returns the optimised curvatures
        in two element list.
        """
        return self._optimized_curvatures
    def RMS(self, screen):
        """
        Method has an OutpuPlane object as its argument.
        Method returns the root mean square radius of the spots formed on an 
        OutputPlane element measured from the position of the centroid ray in
        the OutputPlane element.
        """
        MS = 0
        for spot in screen._screen_pos:
            
            centroid = screen._screen_pos[0]
                    
            mean_squared = (((spot[0]-centroid[0]) * (spot[0]-centroid[0])
                            + (spot[1]-centroid[1]) * (spot[1]-centroid[1])) 
                            / len(screen._screen_pos))
            
            MS += mean_squared
                
        RMS = sp.sqrt(MS)
        
        return RMS
    
    def GEO(self, screen):
        """
        Method has an OutputPlane object as its argument.
        Methof returns the maximum radius of a spot measured from the centroid
        ray in the OutputPlane element.
        """
        radii = []
        for spot in screen._screen_pos:
            
            centroid = screen._screen_pos[0]
            
            radius = (sp.sqrt((spot[0]-centroid[0]) * (spot[0]-centroid[0])
                            + (spot[1]-centroid[1]) * (spot[1]-centroid[1])))
            
            radii.append(radius)
        
        return max(radii)
            
        
