# Ray-tracing
SUBMITTED FILES:

ray.py: (main class) contains Ray class
ray_bundles.py: contains Uniform class
opticalelements.py: contains OpticalElement, SphericalRefraction and 
                    OutputPlane class
optimizer.py: contains Optimization class
test_script.py: contains conducted tests of the code.
plots_final.py: contains final figures and calculations (see last 2 sections
                in this file for important information)
README.txt
______________________________________________________________________________
GETTING STARTED WITH THE CODE

    Start by creating a ray from the Ray class specifying its initial position
and direction. The direction must have a positive z component in order for 
other methods in other class to work with this ray.

    Continue by creating a spherical optical element (a lens) from the
SphericalRefraction class specifying the initial parameters: position of 
surface 1, position of surface2, their respective curvatures, the refractive 
index of the medium surrounding the lens, the refractive index of the lens 
itself and the aperture radius of the lens. 

Note: curvature is defined as in project script. positive for z>z_0 and
negative for z<z_0.

Also note: surface 1 is the first intercepted side of the lens making 
surface 2 the other side of the lens. 

Also note: if the curvature is positive, the positions of the lens 
(z_0_1 or z_0_2), is the lowest point (with respect to z axis) on the 
spherical lens and is the highest point (with respect to z axis) is the 
curvature is negative.
    
    After this you can create an imaging screen using the OutputPlane class
specifying where on the z axis the plane (which is parallel to the xy plane)
is. These spots can be accessed by their components using the screen.spots() 
method or as a list of nested position arrays from the attribute _screen_pos
or the current ray to last be imaged on the screen can be accessed from the
attribute _screen_point. 

    Then you can propagate the ray through both of the spherical element and 
output plane.
    
    More lenses can be created as well as more output planes and a ray 
may propagate through each if the propagation method is called in 
order on the code.

    Alternatively. you may iterate inside a for loop by accessing every 
beam in a uniform collimated bundle of rays formed in the Uniform class. 
Carry out each of the methods above on each ray within this for loop.
The logic to follow always is that all methods work for individual rays so 
for any group of rays one can use a for loop to iterate over each ray.

    A good example is given in the last cell in the plots_final.py file.
______________________________________________________________________________
FLAWS OF THE CODE 

The code is designed create entire lenses, i.e. lenses with two spherical 
surfaces. To create a single spherical lens hence the code must be modified by
COMMENTING OUT a single line in the opticalelements.py file.

This line is in the SphericalRefraction class, in the propagate_ray() method
and is specifically 

ray.append(self.intercept(ray, self.z_0_2), self.Snellslaw(ray, self.z_0_2)).

It is hence recommended to create a new file with a new name that is identical
to to the opticalelements.py except for this line which can be omitted.
This file can then be imported when one wants to investigate single surface
spherical elements.

______________________________________________________________________________
MODIFICATION MADE TO CODE ON test_script.py and plots_final.py

The test_script.py has a section commented out for lens optimization testing
due to long running times.

For the plots_final.py, the lens optimization plot has been modified such that
it will take about 40 seconds to run (minimum possible9 but this is NOT the 
plot presented officially.

Note: This change makes the plot for the uniform bundle at an angle to overlap
both curves.
To make this plot the number of rays in the bundle, n, should be 19 rather 
than 7 and the number of steps in the plane_z = sp.linspace() should be 40 
rather than 6.

Warning: this setup takes a longer amount of time than preferable.
