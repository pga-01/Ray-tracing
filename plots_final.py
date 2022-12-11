from ray import Ray
from optimizer import Optimization
from ray_bundles import Bundle, Uniform
from opticalelements import SphericalRefraction, OutputPlane
from matplotlib.patches import Circle, PathPatch
import scipy as sp
from scipy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.transforms import Affine2D

#%%

"""TASK9"""
"""
PLEASE REFER TO README FILE TO PROPERLY RUN THIS PLOT""
"""

s9 = SphericalRefraction((0, 0, 100), (0, 0, 200), 0.03, -0.03, 1, 1.5, 1/0.03,
                         1,1,1,1)

opt_axis_task9 = Ray((0, 0, 0), (0, 0, 1))
ray0 = Ray((0, 0.01, 0), (0, 0, 1))
ray1 = Ray((-8, 0, 0), (0, 0, 1))
ray2 = Ray((0, 10, 0), (0, 0, 1))
ray3 = Ray((0, -6, 0), (0, 0, 1))

s9.propagate_ray(opt_axis_task9)
s9.propagate_ray(ray0)
s9.propagate_ray(ray1)
s9.propagate_ray(ray2)
s9.propagate_ray(ray3)

paraxial_focus_task9 = ray0.intersection(opt_axis_task9)

p9 = OutputPlane((0, 0, 272))

p9.propagate_ray(opt_axis_task9)
p9.propagate_ray(ray0)
p9.propagate_ray(ray1)
p9.propagate_ray(ray2)
p9.propagate_ray(ray3)
# p9.propagate_ray(ray4)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(azim = 30, elev = 23)


ax.plot(ray0.coordinates()[0], ray0.coordinates()[2], ray0.coordinates()[1], 
        label = "p: [0, 0.01], 0], k: [0, 0, 1]", linewidth = 1)
ax.plot(ray1.coordinates()[0], ray1.coordinates()[2], ray1.coordinates()[1], 
        label = "p: [-8, 0, 0], k: [0, 0, 1]")
ax.plot(ray2.coordinates()[0], ray2.coordinates()[2], ray2.coordinates()[1], 
        label = "p: [0, 10, 0], k: [0, 0, 1")
ax.plot(ray3.coordinates()[0], ray3.coordinates()[2], ray3.coordinates()[1], 
        label = "p: [0, -6, 0], k: [0, 0, 1]")


ax.scatter(p9.screen_spots()[0], p9.screen_spots()[2], p9.screen_spots()[1], 
            color = "black")


x_patch = sp.array([[-20, 20]])
y_patch = sp.array([[min(p9.screen_spots()[2]), min(p9.screen_spots()[2])]])
z_patch = sp.array([[-9, -9], [9, 9]])
ax.plot_surface(x_patch, y_patch, z_patch, alpha = 0.1, color = "b")

ax.set_xlabel("x (mm)", labelpad = 5)
ax.set_ylabel("z (mm)", labelpad = 5.5)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("y (mm)", labelpad = 1, rotation = 90)

ax.tick_params(axis="z", pad = 3, labelsize = 10)
ax.tick_params(axis= "y", pad = 2, labelsize = 10)
ax.tick_params(axis = "x", pad = 3, labelsize = 10)

ax.legend(title = "Initial position \nand direction of ray",fontsize = 9, 
          loc = "best")

plt.tight_layout()
plt.savefig("single_spherical_lens_Task_9.jpeg", dpi=400)
plt.show()


print("The paraxial focus is,", paraxial_focus_task9)
#%%
"TASK 9 TEST CASES"
"""
PLEASE REFER TO README FILE TO PROPERLY RUN THIS PLOT""
"""
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(azim = 0, elev = 0)

opt_axis_task9 = Ray((0, 0, 0), (0, 0, 1))
ray1 = Ray((0, 5, 0), (0, 0, 1))
ray4 = Ray((0, 5, 0), (0, -5, 400/3)) #through centre of curvature
ray5 = Ray((0, 0, 0), (0, 5, 400/3))

s9.propagate_ray(opt_axis_task9)
s9.propagate_ray(ray1)
s9.propagate_ray(ray4)
s9.propagate_ray(ray5)

focal = ray1.intersection(ray4)


p9 = OutputPlane((0, 0, focal[2]))
p9_opt_axis = OutputPlane((0, 0, 420))

p9_opt_axis.propagate_ray(opt_axis_task9)
p9.propagate_ray(ray1)
p9.propagate_ray(ray4)
p9.propagate_ray(ray5)




ax.plot(opt_axis_task9.coordinates()[0], opt_axis_task9.coordinates()[2], 
        opt_axis_task9.coordinates()[1],  linestyle = "--", linewidth = 1, 
        color = "k")
ax.plot(ray1.coordinates()[0], ray1.coordinates()[2], ray1.coordinates()[1], 
        label = "p: [0, 5, 0], k: [0, 0, 1]")
ax.plot(ray4.coordinates()[0], ray4.coordinates()[2], ray4.coordinates()[1], 
        label = "p: [0, 5, 0], k: [0, -5, 400/3]")
ax.plot(ray5.coordinates()[0], ray5.coordinates()[2], ray5.coordinates()[1], 
        label = "p: [0, 0, 0], k: [7, 5, 400/3]")

ax.set_ylabel("z (mm)", labelpad = 5.5)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("y (mm)", labelpad = 1, rotation = 90)

ax.tick_params(axis="z", pad = 3, labelsize = 10)
ax.tick_params(axis= "y", pad = 2, labelsize = 10)
ax.set_xticklabels([])
ax.legend(title = "Initial position and \ndirection of ray",fontsize = 7, 
           loc = 9, bbox_to_anchor = (0.7, 0.99))

plt.savefig("task_9_test_cases.jpeg", dpi = 400)
plt.show()

#%%
"""TASK 12"""
"""
PLEASE REFER TO README FILE TO PROPERLY RUN THIS PLOT""
"""

bundle12 = Uniform(r = 10, n = 91, bundle_z = 0)
s12_1 = SphericalRefraction((0, 0, 100), (0, 0, 120), 0.03, -0.03, 1, 1.5, 
                            1/0.03)
p12_1 = OutputPlane((0, 0, 200))

fig = plt.figure()
ax = fig.gca(projection='3d')
# ax = fig.add_subplot(2, 2, 1, projection = "3d")
ax.view_init(azim = 30, elev = 20)

for ray_i in bundle12.spots():
    beam = Ray(ray_i, (0,0, 1))
    print("THIS ISS", beam.p())
    s12_1.propagate_ray(beam)
    p12_1.propagate_ray(beam)
    print(p12_1._screen_point)
   
    print("HERE IS", beam.coordinates()[0])
    
    
    ax.plot(beam.coordinates()[0], 
            beam.coordinates()[2], 
            beam.coordinates()[1], 
            color = "mediumblue", linewidth = 0.5)
    
circle11_4 = Circle((0, 0), 10+0.5, alpha = 0.1, color = "navy")
ax.add_patch(circle11_4)

art3d.pathpatch_2d_to_3d(circle11_4, z = 0, zdir = "y")

x_patch = sp.array([[-8, 8]])
y_patch = sp.array([[min(p12_1.screen_spots()[2]), 
                     min(p12_1.screen_spots()[2])]])
z_patch = sp.array([[-8, -8], [8, 8]])

ax.plot_surface(x_patch, y_patch, z_patch, alpha = 0.09, color = "k")

ax.set_xlabel("x (mm)", labelpad = 5)
ax.set_ylabel("z (mm)", labelpad = 5)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("y (mm)", labelpad = 1, rotation = 90)

ax.set_xlim(-12, 12)
ax.set_zlim(-12, 12)

ax.tick_params(axis="z", pad = 3, labelsize = 8)
ax.tick_params(axis= "y", pad = 2, labelsize = 8)
ax.tick_params(axis = "x", pad = 3, labelsize = 8)
plt.savefig("collimated_beam_task12.jpeg", dpi = 400)
plt.show()
#%%
"""
PLEASE REFER TO README FILE TO PROPERLY RUN THIS PLOT""
"""
fig = plt.figure(figsize = [5,5])
    
for ray_i in bundle12.spots():
    beam = Ray(ray_i, (0,0,1))
   
    s12_1.propagate_ray(beam)
    p12_1.propagate_ray(beam)
    screen_x = p12_1._screen_point[0]
    
    screen_y = p12_1._screen_point[1]
    plt.scatter(screen_x, screen_y, color = "mediumblue", s = 8 )

plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.savefig("spot_diagram_task12.jpeg", dpi = 400)
plt.show()


#%%
"""TASK 15"""
"""PERFORMANCE PLOT CURV FIRST"""

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(azim = 0, elev = 0)


s15_1 = SphericalRefraction((0, 0, 70), (0, 0, 75), 0.03, 0, 1, 1.5168
                            
                            , 30)
opt_axis = Ray((0, 0, 0), (0, 0, 1))

ray15_m1_testing = Ray((0, s15_1._a_R - 1e-13, 0), (0, 0, 1))

#Marginal rays
ray15_m1 = Ray((0, s15_1._a_R - 1e-13, 0), (0, 0, 1))
ray15_m2 = Ray((0, -s15_1._a_R + 1e-13, 0), (0, 0, 1))

#Paraxial rays
ray15_p1 = Ray((0, 0.000001, 0), (0, 0, 1))
ray15_p2 = Ray((0, -0.000001, 0), (0, 0, 1))

#Best focus rays
ray15_b1 = Ray((0,(s15_1._a_R - 1e-13)/sp.sqrt(2), 0), (0, 0, 1))
ray15_b2 = Ray((0,(-s15_1._a_R + 1e-13)/sp.sqrt(2), 0), (0, 0, 1))


s15_1.propagate_ray(opt_axis)

s15_1.propagate_ray(ray15_m1_testing)

s15_1.propagate_ray(ray15_m1)
s15_1.propagate_ray(ray15_m2)
s15_1.propagate_ray(ray15_p1)
s15_1.propagate_ray(ray15_p2)
s15_1.propagate_ray(ray15_b1)
s15_1.propagate_ray(ray15_b2)

marginal_focus_1 = ray15_m1.intersection(ray15_m2)
paraxial_focus_1 = ray15_p1.intersection(ray15_p2)
best_focus_1 = ray15_b1.intersection(opt_axis)

CofLC_pos_1 = ray15_p1.intersection(ray15_m2)

mag_1 = paraxial_focus_1[2]/(paraxial_focus_1[2] - 70) #since 70mm is the 
#object distance in this case


print("The marginal focus (convex first) is at:", marginal_focus_1)
print("The paraxial focus (convex first) is at:", paraxial_focus_1)
print("The best focus (convex first) is at:", best_focus_1)
print("The magnification (convex first) is:", mag_1)

p_opt_axis = OutputPlane((0, 0, 170))

p15_1_mfocus = OutputPlane((0, 0, marginal_focus_1[2]))
p15_1_pfocus = OutputPlane((0, 0, paraxial_focus_1[2]))
p15_1_bfocus = OutputPlane((0, 0, best_focus_1[2]))

#Propagate esch type of ray to its respective focal point
p_opt_axis.propagate_ray(opt_axis)

p15_1_pfocus.propagate_ray(ray15_m1_testing)
p15_1_mfocus.propagate_ray(ray15_m1)
p15_1_mfocus.propagate_ray(ray15_m2)
p15_1_pfocus.propagate_ray(ray15_p1)
p15_1_pfocus.propagate_ray(ray15_p2)
p15_1_bfocus.propagate_ray(ray15_b1)
p15_1_bfocus.propagate_ray(ray15_b2)






ax.plot(ray15_m1.coordinates()[0], 
        ray15_m1.coordinates()[2], 
        ray15_m1.coordinates()[1], 
        label = "Marginal rays", color = "red", linewidth = 0.6)
ax.plot(ray15_m2.coordinates()[0], 
        ray15_m2.coordinates()[2], 
        ray15_m2.coordinates()[1],
        color = "red", linewidth = 0.6)
ax.plot(ray15_p1.coordinates()[0], 
        ray15_p1.coordinates()[2], 
        ray15_p1.coordinates()[1], 
        label = "Paraxial rays", color = "blue", linewidth = 0.6)
ax.plot(ray15_p2.coordinates()[0], 
        ray15_p2.coordinates()[2], 
        ray15_p2.coordinates()[1], 
        color = "blue", linewidth = 0.2)
ax.plot(ray15_b1.coordinates()[0], 
        ray15_b1.coordinates()[2], 
        ray15_b1.coordinates()[1], 
        label = "Best focus rays", color = "green", linewidth = 0.6)
ax.plot(ray15_b2.coordinates()[0], 
        ray15_b2.coordinates()[2], 
        ray15_b2.coordinates()[1], 
        color = "green", linewidth = 0.6)


# ax.set_xlabel("x (mm)", labelpad = 5)
ax.set_ylabel("z (mm)", labelpad = 5.5)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("y (mm)", labelpad = 1, rotation = 90)
ax.set_xticklabels([])


ax.tick_params(axis="z", pad = 3, labelsize = 10)
ax.tick_params(axis= "y", pad = 2, labelsize = 10)
ax.tick_params(axis = "x", pad = 3, labelsize = 10)

plt.legend(fontsize = 9, bbox_to_anchor = (0.7, 0.8), loc = 2)

plt.savefig("plano_convex_lens_performance_Task15_1.jpeg", dpi = 400)
plt.show()



long_sa_1 = abs(paraxial_focus_1[2] - marginal_focus_1[2])
trans_sa_1 = sp.sqrt(ray15_m1_testing.p()[0]*ray15_m1_testing.p()[0] 
                          + ray15_m1_testing.p()[1]*ray15_m1_testing.p()[1])
CofLC_r_1 = sp.sqrt(CofLC_pos_1[0]*CofLC_pos_1[0] 
                    + CofLC_pos_1[1]*CofLC_pos_1[1])

print("The longitudional spherical aberration (convex first) is:", long_sa_1)
print("The transverse spherical aberration (convex first) is:", trans_sa_1)
print("The radius of the circle of least confusion is (convex first) is:"
      , CofLC_r_1)



#%%
"""TASK 15 PERFORMANCE PLOT PLANO FIRST"""


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(azim = 0, elev = 0)



s15_2 = SphericalRefraction((0, 0, 70), (0, 0, 75), 0, -0.03, 1, 1.5168
                            
                            , 30)
#Optical axis
opt_axis = Ray((0, 0, 0), (0, 0, 1))

ray15_m1_testing = Ray((0, s15_2._a_R - 1e-13, 0), (0, 0, 1))

#Marginal rays
ray15_m1 = Ray((0, s15_2._a_R - 1e-13, 0), (0, 0, 1))
ray15_m2 = Ray((0, -s15_2._a_R + 1e-13, 0), (0, 0, 1))

#Paraxial rays
ray15_p1 = Ray((0, 0.000001, 0), (0, 0, 1))
ray15_p2 = Ray((0, -0.000001, 0), (0, 0, 1))

#Best focus rays
ray15_b1 = Ray((0,(s15_2._a_R - 1e-13)/sp.sqrt(2), 0), (0, 0, 1))
ray15_b2 = Ray((0,(-s15_2._a_R + 1e-13)/sp.sqrt(2), 0), (0, 0, 1))

s15_2.propagate_ray(opt_axis)

s15_2.propagate_ray(ray15_m1_testing)

s15_2.propagate_ray(ray15_m1)
s15_2.propagate_ray(ray15_m2)
s15_2.propagate_ray(ray15_p1)
s15_2.propagate_ray(ray15_p2)
s15_2.propagate_ray(ray15_b1)
s15_2.propagate_ray(ray15_b2)

marginal_focus_2 = ray15_m1.intersection(ray15_m2)
paraxial_focus_2 = ray15_p1.intersection(ray15_p2)
best_focus_2 = ray15_b1.intersection(ray15_b2)
mag_2 = paraxial_focus_2[2]/(paraxial_focus_2[2] - 70) #since 70mm is the 
#object distance

CofLC_pos_2 = ray15_p1.intersection(ray15_m2)



print("The marginal focus (plano first) is at:", marginal_focus_2)
print("The paraxial focus (plano first)is:", paraxial_focus_2)
print("The best focus (plano first) is:", best_focus_2)
print("The magnification (plano first) is:", mag_2)

p_opt_axis = OutputPlane((0, 0, 170))

p15_2_mfocus = OutputPlane((0, 0, marginal_focus_2[2]))
p15_2_pfocus = OutputPlane((0, 0, paraxial_focus_2[2]))
p15_2_bfocus = OutputPlane((0, 0, best_focus_2[2]))

#Propagate esch type of ray to its respective focal point
p_opt_axis.propagate_ray(opt_axis)

p15_2_pfocus.propagate_ray(ray15_m1_testing)
p15_2_mfocus.propagate_ray(ray15_m1)
p15_2_mfocus.propagate_ray(ray15_m2)
p15_2_pfocus.propagate_ray(ray15_p1)
p15_2_pfocus.propagate_ray(ray15_p2)
p15_2_bfocus.propagate_ray(ray15_b1)
p15_2_bfocus.propagate_ray(ray15_b2)


ax.plot(ray15_m1.coordinates()[0], 
        ray15_m1.coordinates()[2], 
        ray15_m1.coordinates()[1], 
        label = "Marginal rays", color = "red", linewidth = 0.6)
ax.plot(ray15_m2.coordinates()[0], 
        ray15_m2.coordinates()[2], 
        ray15_m2.coordinates()[1],
        color = "red", linewidth = 0.6)
ax.plot(ray15_p1.coordinates()[0], 
        ray15_p1.coordinates()[2], 
        ray15_p1.coordinates()[1], 
        label = "Paraxial rays", color = "blue", linewidth = 0.6)
ax.plot(ray15_p2.coordinates()[0], 
        ray15_p2.coordinates()[2], 
        ray15_p2.coordinates()[1], 
        color = "blue", linewidth = 0.2)
ax.plot(ray15_b1.coordinates()[0], 
        ray15_b1.coordinates()[2], 
        ray15_b1.coordinates()[1], 
        label = "Best focus rays", color = "green", linewidth = 0.6)
ax.plot(ray15_b2.coordinates()[0], 
        ray15_b2.coordinates()[2], 
        ray15_b2.coordinates()[1], 
        color = "green", linewidth = 0.6)


ax.set_ylabel("z (mm)", labelpad = 5.5)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("y (mm)", labelpad = 1, rotation = 90)

ax.tick_params(axis="z", pad = 3, labelsize = 8)
ax.tick_params(axis= "y", pad = 2, labelsize = 8)
ax.tick_params(axis = "x", pad = 3, labelsize = 8)
ax.set_xticklabels([])

plt.legend(fontsize = 9, bbox_to_anchor = (0.7, 0.8), loc = 2)

plt.savefig("plano_convex_lens_performance_Task15_2.jpeg", dpi = 400)
plt.show()



long_sa_2 = abs(paraxial_focus_2[2] - marginal_focus_2[2])
trans_sa_2 = sp.sqrt(ray15_m1_testing.p()[0]*ray15_m1_testing.p()[0] 
                     + ray15_m1_testing.p()[1]*ray15_m1_testing.p()[1])
CofLC_r_2 = sp.sqrt(CofLC_pos_2[0]*CofLC_pos_2[0] 
                    + CofLC_pos_2[1]*CofLC_pos_2[1])

print("The longitudional spherical aberration (plano first) is:", long_sa_2)
print("The transverse spherical aberration (plano first) is:", trans_sa_2)
print("The radius of the circle of least confusion is (plano first) is:", 
      CofLC_r_2)
#%%
"""TASK 15"""
"""PLANO CONVEX LENS CONVEX FIRST"""


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(azim = 0, elev = 20)

bundle15 = Uniform(r = 10, n = 19, bundle_z = 35)


s15_1 = SphericalRefraction((0, 0, 70), (0, 0, 75), 0.03, 0, 1, 1.5168
                            , 30)
p15_1 = OutputPlane((0, 0, 160))

for ray_i in bundle15.spots():
    beam = Ray(ray_i, (0,0, 1))
    s15_1.propagate_ray(beam)
    p15_1.propagate_ray(beam)
    
    ax.plot(beam.coordinates()[0], 
            beam.coordinates()[2], 
            beam.coordinates()[1], 
            color = "mediumblue", linewidth = 0.6)
ax.scatter(p15_1.screen_spots()[0], 
           p15_1.screen_spots()[2], 
           p15_1.screen_spots()[1], 
           color = "k", s = 3)
    

circle15 = Circle((0, 0), bundle15._r+0.5, alpha = 0.1, color = "navy")
ax.add_patch(circle15)
art3d.pathpatch_2d_to_3d(circle15, z = 35, zdir = "y")

x_patch = sp.array([[-20, 20]])
y_patch = sp.array([[min(p15_1.screen_spots()[2]),
                     min(p15_1.screen_spots()[2])]])
z_patch = sp.array([[-15, -15], [56, 56]])
ax.plot_surface(x_patch, y_patch, z_patch, alpha = 0.09, color = "k")

ax.set_xlabel("x (mm)", labelpad = 5)
ax.set_ylabel("z (mm)", labelpad = 5.5)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("y (mm)", labelpad = 1, rotation = 90)

ax.tick_params(axis="z", pad = 3, labelsize = 8)
ax.tick_params(axis= "y", pad = 2, labelsize = 8)
ax.tick_params(axis = "x", pad = 3, labelsize = 8)

s15_1 = SphericalRefraction((0, 0, 70), (0, 0, 75), 0.03, 0, 1, 1.5168
                            , 30)
p15_1 = OutputPlane((0, 0, 160))


bundle15_2 = Uniform(r = 10, n = 19 , bundle_z = 35)

bundle15_2.translate((0, -15, 35))

for ray_i in bundle15_2.spots():
    beam = Ray(ray_i, (0, 15, 35))
    s15_1.propagate_ray(beam)
    p15_1.propagate_ray(beam)
    
    ax.plot(beam.coordinates()[0], 
            beam.coordinates()[2], 
            beam.coordinates()[1], 
            color = "red", linewidth = 0.3)
    
ax.scatter(p15_1.screen_spots()[0], 
            p15_1.screen_spots()[2], 
            p15_1.screen_spots()[1], 
            color = "k", linewidth = 0.2, marker = ".")


circle15_2 = Circle((0, -15), bundle15_2._r+0.5, alpha = 0.1, color = "red")
ax.add_patch(circle15_2)
art3d.pathpatch_2d_to_3d(circle15_2, z = 35, zdir = "y")


ax.set_xlabel("x (mm)", labelpad = 5)
ax.set_ylabel("z (mm)", labelpad = 5.5)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel("y (mm)", labelpad = 1, rotation = 90)

ax.tick_params(axis="z", pad = 3, labelsize = 8)
ax.tick_params(axis= "y", pad = 2, labelsize = 8)
ax.tick_params(axis = "x", pad = 3, labelsize = 8)
plt.savefig("plano_convex_lens_1_b_and_r.jpeg", dpi = 400)
plt.show()
#%%
fig = plt.figure(figsize = [5,5])

bundle15 = Uniform(r = 10, n =91, bundle_z = 35)
p15_1_focus = OutputPlane(((0, 0, paraxial_focus_1[2])))

    
for ray_i in bundle15.spots():
    beam = Ray(ray_i, (0,0,1))
   
    s15_1.propagate_ray(beam)
    p15_1_focus.propagate_ray(beam)
    screen_x = p15_1_focus._screen_point[0]
    screen_y = p15_1_focus._screen_point[1]
    plt.scatter(screen_x, screen_y, color = "mediumblue", s = 8 )

plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.xticks(fontsize = 10)
plt.savefig("spot_diagram_task15_1_b_pfocus.jpeg", dpi = 400)
plt.show()

#RMS calculation
opt_plano_convex_1_b = Optimization()
RMS_spot_1_b = opt_plano_convex_1_b.RMS(p15_1_focus)
GEO_spot_1_b = opt_plano_convex_1_b.GEO(p15_1_focus)
print("The RMS radius is:", RMS_spot_1_b)
print("The geometrical radius is:", GEO_spot_1_b)

#%%
fig = plt.figure(figsize = [5,5])

bundle15_2 = Uniform(r = 10, n = 91 , bundle_z = 35)
bundle15_2.translate((0, -30, 35))

p15_1_focus = OutputPlane(((0, 0, paraxial_focus_1[2])))

for ray_i in bundle15_2.spots():
    beam = Ray(ray_i, (0, 30, 35))
   
    s15_1.propagate_ray(beam)
    p15_1_focus.propagate_ray(beam)
    screen_x = p15_1_focus._screen_point[0]
    
    screen_y = p15_1_focus._screen_point[1]
    plt.scatter(screen_x, screen_y, color = "red", s = 8 )

plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.xticks(fontsize = 9)
plt.savefig("spot_diagram_task15_1_r_pfocus.jpeg", dpi = 400)
plt.show()

#RMS calculation
opt_plano_convex_1_r = Optimization()
RMS_spot_1_r = opt_plano_convex_1_r.RMS(p15_1_focus)
GEO_spot_1_r = opt_plano_convex_1_r.GEO(p15_1_focus)
print("The RMS radius is:", RMS_spot_1_r)
print("The geometrical radius is:", GEO_spot_1_r)


#%%
fig = plt.figure(figsize = [5,5])

bundle15 = Uniform(r = 10, n = 91, bundle_z = 35)
p15_1_focus = OutputPlane(((0, 0, paraxial_focus_2[2])))

    
for ray_i in bundle15.spots():
    beam = Ray(ray_i, (0,0,1))
   
    s15_2.propagate_ray(beam)
    p15_1_focus.propagate_ray(beam)
    screen_x = p15_1_focus._screen_point[0]
    
    screen_y = p15_1_focus._screen_point[1]
    plt.scatter(screen_x, screen_y, color = "mediumblue", s = 8 )

plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.savefig("spot_diagram_task15_2_b_pfocus.jpeg", dpi = 400)
plt.show()

# RMS calculation
opt_plano_convex_2_b = Optimization()
RMS_spot_2_b = opt_plano_convex_2_b.RMS(p15_1_focus)
GEO_spot_2_b = opt_plano_convex_2_b.GEO(p15_1_focus)
print("The RMS radius is:", RMS_spot_2_b)
print("The geometrical radius is:", GEO_spot_2_b)
#%%
fig = plt.figure(figsize = [5,5])

bundle15_2 = Uniform(r = 10, n = 91 , bundle_z = 35)
bundle15_2.translate((0, -30, 35))

p15_1_focus = OutputPlane(((0, 0, paraxial_focus_2[2])))

for ray_i in bundle15_2.spots():
    beam = Ray(ray_i, (0, 30, 35))
   
    s15_2.propagate_ray(beam)
    p15_1_focus.propagate_ray(beam)
    screen_x = p15_1_focus._screen_point[0]
    screen_y = p15_1_focus._screen_point[1]
    plt.scatter(screen_x, screen_y, color = "red", s = 8 )

plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.savefig("spot_diagram_task15_2_r_pfocus.jpeg", dpi = 400)
plt.show()

# RMS calculation
opt_plano_convex_2_r = Optimization()
RMS_spot_2_r = opt_plano_convex_2_r.RMS(p15_1_focus)
GEO_spot_2_r = opt_plano_convex_2_r.GEO(p15_1_focus)
print("The RMS radius is:", RMS_spot_2_r)
print("The geometrical radius is:", GEO_spot_2_r)
#%%
"""RMS vs Z OPTIMIZATION OF LENS"""

"PLANO CONVEX, CONVEX FIRST"
"PARALLEL RAYS"

bundle15 = Uniform(r = 10, n = 37, bundle_z = 35)

s15_1 = SphericalRefraction((0, 0, 70), (0, 0, 75), 0.03, 0, 1, 1.5168
                            , 30)
p15_1 = OutputPlane((0, 0, 160))

opt_RMS_fig = Optimization()
RMS_radii_1_b = []

plane_z = sp.linspace(110, 140, 100, endpoint = True)
for z in plane_z:
    plane = OutputPlane((0, 0, z))
    for ray_i in bundle15.spots():
        beam = Ray(ray_i, (0,0, 1))
        s15_1.propagate_ray(beam)
        
        
        plane.propagate_ray(beam)
        
    RMS_1_b = opt_RMS_fig.RMS(plane)
    RMS_radii_1_b.append(RMS_1_b)
    
#%%
"PLANO CONVEX, CONVEX FIRST"
"ANGLED RAYS"

bundle15_2 = Uniform(r = 10, n = 37 , bundle_z = 35)

bundle15_2.translate((0, -15, 35))

s15_1 = SphericalRefraction((0, 0, 70), (0, 0, 75), 0.03, 0, 1, 1.5168
                            , 30)

opt_RMS_fig = Optimization()
RMS_radii_1_r = []

plane_z = sp.linspace(110, 140, 100, endpoint = True)
for z in plane_z:
    plane = OutputPlane((0, 0, z))
    for ray_i in bundle15_2.spots():
        beam = Ray(ray_i, (0, 15, 35))
        s15_1.propagate_ray(beam)
        
        
        plane.propagate_ray(beam)
        
    RMS_1_r = opt_RMS_fig.RMS(plane)
    RMS_radii_1_r.append(RMS_1_r)
#%%
"PLANO CONVEX, planar FIRST"
"PARALLEL RAYS"

bundle15 = Uniform(r = 10, n = 37, bundle_z = 35)

s15_2 = SphericalRefraction((0, 0, 70), (0, 0, 75), 0, -0.03, 1, 1.5168
                            , 30)

opt_RMS_fig = Optimization()
RMS_radii_2_b = []
plane_z = sp.linspace(110, 140, 100, endpoint = True)
for z in plane_z:
    plane = OutputPlane((0, 0, z))
    for ray_i in bundle15.spots():
        beam = Ray(ray_i, (0,0, 1))
        s15_2.propagate_ray(beam)
        
        plane.propagate_ray(beam)
        
    RMS = opt_RMS_fig.RMS(plane)
    RMS_radii_2_b.append(RMS)
#%%
"PLANO CONVEX, planar FIRST"
"ANGLED RAYS"

bundle15_2 = Uniform(r = 10, n = 37 , bundle_z = 35)

bundle15_2.translate((0, -15, 35))

s15_2 = SphericalRefraction((0, 0, 70), (0, 0, 75), 0, -0.03, 1, 1.5168
                            , 30)

opt_RMS_fig = Optimization()
RMS_radii_2_r = []

plane_z = sp.linspace(110, 140, 100, endpoint = True)
for z in plane_z:
    plane = OutputPlane((0, 0, z))
    for ray_i in bundle15_2.spots():
        beam = Ray(ray_i, (0, 15, 35))
        s15_2.propagate_ray(beam)
        
        
        plane.propagate_ray(beam)
        
    RMS_2_r = opt_RMS_fig.RMS(plane)
    RMS_radii_2_r.append(RMS_2_r)
#%%
from scipy.optimize import curve_fit
def abs_func(x, a, phase):
    return a*abs(x-phase)
#Initial guesses for 1_b
iphase_1_b = 134
ia_1_b = 0.2
guesses_1_b = [ia_1_b, iphase_1_b]

#Curve_fitting for 1_b
fit_1_b = curve_fit(abs_func, plane_z, RMS_radii_1_b, guesses_1_b)
data_fit_1_b = abs_func(plane_z, *fit_1_b[0])

psigma_1_b = sp.sqrt(sp.diag(fit_1_b[1]))


#Polyfit for 1_r
fit_1_r, cov_1_r = sp.polyfit(plane_z, RMS_radii_1_r, 8, cov = True)

sig_slope_1_r = sp.sqrt(cov_1_r[0,0]) # Uncertainty in the slope
sig_inter_1_r = sp.sqrt(cov_1_r[1,1]) # Uncertainty in the intercept

print('The slope is: %.3e +/- %.3e' %(fit_1_r[0],sig_slope_1_r))
print('Intercept = %.3e +/- %.3e' %(fit_1_r[1],sig_inter_1_r))

pRMS_1_r = sp.poly1d(fit_1_r)

#Polyfit for 2_b
fit_2_b, cov_2_b = sp.polyfit(plane_z, RMS_radii_2_b, 6, cov = True)

sig_slope_2_b = sp.sqrt(cov_2_b[0,0]) # Uncertainty in the slope
sig_inter_2_b = sp.sqrt(cov_2_b[1,1]) # Uncertainty in the intercept

print('The slope is: %.3e +/- %.3e' %(fit_2_b[0],sig_slope_2_b))
print('Intercept = %.3e +/- %.3e' %(fit_2_b[1],sig_inter_2_b))

pRMS_2_b = sp.poly1d(fit_2_b)

#Polyfit for 2_r
fit_2_r, cov_2_r = sp.polyfit(plane_z, RMS_radii_2_r, 5, cov = True)

sig_slope_2_r = sp.sqrt(cov_2_r[0,0]) # Uncertainty in the slope
sig_inter_2_r = sp.sqrt(cov_2_r[1,1]) # Uncertainty in the intercept

print('The slope is: %.3e +/- %.3e' %(fit_2_r[0],sig_slope_2_r))
print('Intercept = %.3e +/- %.3e' %(fit_2_r[1],sig_inter_2_r))

pRMS_2_r = sp.poly1d(fit_2_r)

#minima
min_RMS_radii_1_b = min(data_fit_1_b)
min_RMS_radii_1_r = min(pRMS_1_r(plane_z))
min_RMS_radii_2_b = min(pRMS_2_b(plane_z))
min_RMS_radii_2_r = min(pRMS_2_r(plane_z))



plt.grid()
plt.title(("RMS radius of ray spots at different positions on "
            "\nthe optical axis past the plano convex lens"))
plt.xlabel(r"$z$ $(mm)$")
plt.ylabel("RMS radius $(mm)$")
# plt.scatter(plane_z, RMS_radii_1_b, marker = ".", s = 9, color = "red")
# plt.scatter(plane_z, RMS_radii_1_r, marker = ".", s = 4 )
# plt.scatter(plane_z, RMS_radii_2_b, marker = ".", s = 8, color = "k")
# plt.scatter(plane_z, RMS_radii_2_r, marker = ".", s = 10 )

plt.plot(plane_z, data_fit_1_b, color = "mediumblue", 
         label = "<--- \nParallel rays" )
plt.plot(plane_z, pRMS_1_r(plane_z), color = "red", 
         label = "<--- \nAngled rays")
plt.plot(plane_z, pRMS_2_b(plane_z), color = "mediumblue", 
         linestyle = "--", label = "---> \nParallel rays")
plt.plot(plane_z, pRMS_2_r(plane_z), color = "red", 
         linestyle = "--", label = "---> \nAngled rays")

plt.legend(loc = 6, bbox_to_anchor = (0.28, 0.77), fontsize = 8.1)

plt.savefig("RMS vs z.jpeg", dpi = 400)
plt.show()
#%%
"""TASK LENS OPTIMIZATION"""

""" INITIALLY MODIFIED TO REDUCE EXCESSIVE RUNTIME, PLEASE REFER TO README
FILE TO ACCESS ACTUAL PRESENTED PLOT. """

"PLANO CONVEX, CONVEX FIRST"
"PARALLEL RAYS"

bundle15 = Uniform(r = 10, n = 7, bundle_z = 35)

s15_1 = SphericalRefraction((0, 0, 70), (0, 0, 75), 0.03, 0, 1, 1.5168
                            , 30)

opt_curv_fig_1_b = Optimization()

plane_z_opt = sp.linspace( 92, 500 , 6, endpoint = True)
for z in plane_z_opt:
    plane = OutputPlane((0, 0, z))
    opt_curv_fig_1_b.optimize(bundle15, (0, 0, 1), s15_1, plane)
    
#%%
""" INITIALLY MODIFIED TO REDUCE EXCESSIVE RUNTIME, PLEASE REFER TO README
FILE TO ACCESS ACTUAL PRESENTED PLOT. """

"PLANO CONVEX, CONVEX FIRST"
"ANGLED RAYS"


bundle15_2 = Uniform(r = 10, n = 7, bundle_z = 35)

bundle15_2.translate((0, -15, 35))

s15_1 = SphericalRefraction((0, 0, 70), (0, 0, 75), 0.03, 0, 1, 1.5168
                            , 30)

opt_curv_fig_1_r = Optimization()

plane_z_opt = sp.linspace( 92, 500, 6, endpoint = True)
for z in plane_z_opt:
    plane = OutputPlane((0, 0, z))
    opt_curv_fig_1_r.optimize(bundle15_2, (0, 15, 35), s15_1, plane)

#%%
""" INITIALLY MODIFIED TO REDUCE EXCESSIVE RUNTIME, PLEASE REFER TO README
FILE TO ACCESS ACTUAL PRESENTED PLOT. """

"PLANO CONVEX, planar FIRST"
"PARALLEL RAYS"

bundle15 = Uniform(r = 10, n = 7, bundle_z = 35)

s15_2 = SphericalRefraction((0, 0, 70), (0, 0, 75), 0.03, 0, 1, 1.5168
                            , 30)

opt_curv_fig_2_b = Optimization()

plane_z_opt = sp.linspace( 92, 500, 6, endpoint = True)
for z in plane_z_opt:
    plane = OutputPlane((0, 0, z))
    opt_curv_fig_2_b.optimize(bundle15, (0, 0, 1), s15_2, plane)

#%%
""" INITIALLY MODIFIED TO REDUCE EXCESSIVE RUNTIME, PLEASE REFER TO README
FILE TO ACCESS ACTUAL PRESENTED PLOT. """

"PLANO CONVEX, planar FIRST"
"ANGLED RAYS"

bundle15_2 = Uniform(r = 10, n = 7, bundle_z = 35)

bundle15_2.translate((0, -15, 35))

s15_2 = SphericalRefraction((0, 0, 70), (0, 0, 75), 0.03, 0, 1, 1.5168
                            , 30)

opt_curv_fig_2_r = Optimization()

plane_z_opt = sp.linspace( 92, 500, 6, endpoint = True)
for z in plane_z_opt:
    plane = OutputPlane((0, 0, z))
    opt_curv_fig_2_r.optimize(bundle15_2, (0, 15, 35), s15_2, plane)

#%%
""" INITIALLY MODIFIED TO REDUCE EXCESSIVE RUNTIME, PLEASE REFER TO README
FILE TO ACCESS ACTUAL PRESENTED PLOT. """

#Extraxt curvatures of surface 1 and 2 from optimized curvatures list
curv1_1_b = sp.array([i[0] for i in opt_curv_fig_1_b.opt_curvatures()])
curv2_1_b = sp.array([i[1] for i in opt_curv_fig_1_b.opt_curvatures()])


curv1_1_r = sp.array([i[0] for i in opt_curv_fig_1_r.opt_curvatures()])
curv2_1_r = sp.array([i[1] for i in opt_curv_fig_1_r.opt_curvatures()])

curv1_2_b = sp.array([i[0] for i in opt_curv_fig_2_b.opt_curvatures()])
curv2_2_b = sp.array([i[1] for i in opt_curv_fig_2_b.opt_curvatures()])

curv1_2_r = sp.array([i[0] for i in opt_curv_fig_2_r.opt_curvatures()])
curv2_2_r = sp.array([i[1] for i in opt_curv_fig_2_r.opt_curvatures()])

# fig, ax_1 = plt.subplots()
# ax_1.scatter(plane_z, curv1_1_b, marker = "+", color = "mediumblue")
# ax_1.scatter(plane_z, curv1_1_r, marker = "+", color = "red")
# ax_1.scatter(plane_z, curv1_2_b, marker = ".")
# ax_1.scatter(plane_z, curv1_2_r, marker = ".")

# ax_2 = ax_1.twinx()
# ax_2.scatter(plane_z, curv2_1_b, marker = "x", color = "mediumblue")
# ax_2.scatter(plane_z, curv2_1_r, marker = "x", color = "red")

#Polyfit for curv1_1_b and curv2_1_b
fit_1_b, cov_1_b = sp.polyfit(plane_z_opt, curv1_1_b, 2, cov = True)

sig_slope_1_b = sp.sqrt(cov_1_b[0,0]) # Uncertainty in the slope
sig_inter_1_b = sp.sqrt(cov_1_b[1,1]) # Uncertainty in the intercept

print('The slope is: %.3e +/- %.3e' %(fit_1_b[0],sig_slope_1_b))
print('Intercept = %.3e +/- %.3e' %(fit_1_b[1],sig_inter_1_b))

pcurv_1_b = sp.poly1d(fit_1_b)

#Polyfit for curv1_1_r and curv2_1_r
fit_1_r, cov_1_r = sp.polyfit(plane_z_opt, curv1_1_r, 2, cov = True)

sig_slope_1_r = sp.sqrt(cov_1_r[0,0]) # Uncertainty in the slope
sig_inter_1_r = sp.sqrt(cov_1_r[1,1]) # Uncertainty in the intercept

print('The slope is: %.3e +/- %.3e' %(fit_1_r[0],sig_slope_1_r))
print('Intercept = %.3e +/- %.3e' %(fit_1_r[1],sig_inter_1_r))

pcurv_1_r = sp.poly1d(fit_1_r)

#minima
min_curvature_1_b = min(pcurv_1_b(plane_z_opt))
min_curvature_1_r = min(pcurv_1_r(plane_z_opt))

#Polyfit for curv2_1_b and curv2_2_b
fit_2_b, cov_2_b = sp.polyfit(plane_z_opt, curv2_1_b, 2, cov = True)

sig_slope_2_b = sp.sqrt(cov_2_b[0,0]) # Uncertainty in the slope
sig_inter_2_b = sp.sqrt(cov_2_b[1,1]) # Uncertainty in the intercept

print('The slope is: %.3e +/- %.3e' %(fit_2_b[0],sig_slope_2_b))
print('Intercept = %.3e +/- %.3e' %(fit_2_b[1],sig_inter_2_b))

pcurv_2_b = sp.poly1d(fit_2_b)

#Polyfit for curv2_1_r and curv2_2_r
fit_2_r, cov_2_r = sp.polyfit(plane_z_opt, curv2_1_r, 2, cov = True)

sig_slope_2_r = sp.sqrt(cov_2_r[0,0]) # Uncertainty in the slope
sig_inter_2_r = sp.sqrt(cov_2_r[1,1]) # Uncertainty in the intercept

pcurv_2_r = sp.poly1d(fit_2_r)

#minima
max_curvature_2_b = max(pcurv_2_b(plane_z_opt))
max_curvature_2_r = max(pcurv_2_r(plane_z_opt))
#Plot fitted data
fig, ax_1_b = plt.subplots()
ax_2_b = ax_1_b.twinx()

ax_1_b.grid()
plt.title("Optimized curvature at different positions on "
            "the optical axis past the \nplano convex lens for a uniform ray " 
            "incident parallel to optical axis.", fontsize = 10)
ax_1_b.set_ylabel(r"Surface 1 curvature ($mm^{-1}$)")
ax_1_b.set_xlabel(r"$z$ ($mm$)")
ax_2_b.set_ylabel(r"Surface 2 curvature ($mm^{-1}$)")
ax_1_b.plot(plane_z_opt, pcurv_1_b(plane_z_opt), color = "mediumblue", 
            label = "Surface 1")
ax_2_b.plot(plane_z_opt, pcurv_2_b(plane_z_opt), color = "mediumblue", 
            linestyle = "--", label = "Surface 2")
ax_1_b.legend(loc = 9, bbox_to_anchor = (0.28, 0.999) )
ax_2_b.legend(loc = 9, bbox_to_anchor = (0.28, 0.902))
plt.tight_layout()
plt.savefig("lens_optimization_fig_b.jpeg", dpi = 400)
plt.show()

fig, ax_1_r = plt.subplots()
ax_2_r = ax_1_r.twinx()

ax_1_r.grid()
plt.title("Optimized curvature at different positions on "
            "the optical axis past the \nplano convex lens for a uniform ray " 
            "incident at an angle to optical axis.", fontsize = 10)
ax_1_r.set_ylabel(r"Surface 1 curvature ($mm^{-1}$)")
ax_1_r.set_xlabel(r"$z$ ($mm$)")
ax_2_r.set_ylabel(r"Surface 2 curvature ($mm^{-1}$)")
ax_1_r.plot(plane_z_opt, pcurv_1_r(plane_z_opt), color = "red",
            label = "Surface 1")
ax_2_r.plot(plane_z_opt, pcurv_2_r(plane_z_opt), color = "red", 
            linestyle = "--", label = "Surface 2")
ax_1_r.legend(loc = 9, bbox_to_anchor = (0.5, 0.999) )
ax_2_r.legend(loc = 9, bbox_to_anchor = (0.5, 0.902))
fig.tight_layout()
plt.savefig("lens_optimization_fig_r.jpeg", dpi = 400)
plt.show()
#%%
"""MULTI LENS SYSTEM EXTENSION"""
"Reference: Furkan E. Sahin. Long-range, High-resolution Camera Optical Design"
"for Assited and Autonomous Driving. Journal: Photonics. 2019;6(73). "
"DOI:https://doi.org/10.3390/photonics6020073."

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(azim = 10, elev = 10)



bundle15 = Uniform(r = 10, n = 19, bundle_z = 45)
bundle_angled_r = Uniform(r = 8.5, n = 19, bundle_z = 45)
bundle_angled_r.translate((0, -5, 45))

plano_convex = SphericalRefraction((0, 0, 60), (0, 0, 63), 0.04, 0, 1, 
                                   1.5168, 1/0.05)
biconcave = SphericalRefraction((0, 0, 66), (0, 0, 69.5), -0.02, 0.02, 1, 
                                1.5, 1/0.05 )
biconvex = SphericalRefraction((0, 0, 74), (0, 0, 76), 0.01, -0.03, 1., 
                               1.5168, 1/0.03)
pos_meniscus = SphericalRefraction((0, 0, 79), (0, 0, 82), 0.05, 0.03, 1, 
                                   1.5168, 1/0.05)
plano_concave = SphericalRefraction((0, 0, 86), (0, 0, 88), -0.01, 0, 1, 
                                    1.5168, 1/0.05)

screen = OutputPlane((0, 0, 108.40960996))

#Optical axis ray
opt_axis = Ray("Gaussian",1.0,(0, 0, 40), (0, 0 ,1),w0=1, f=1)

#Paraxial ray
ray_p = Ray("Gaussian",1.0,(0, 0.00001, 40), (0, 0, 1),w0=1, f=1)

#Marginal ray
ray_m = Ray("Gaussian",1.0,(0, 11.0782546, 40 ), (0, 0, 40),w0=1, f=1)#y value found from testing

#Best focus ray
ray_bf = Ray("Gaussian",1.0,(0, 11.0782546/sp.sqrt(2), 40 ), (0, 0, 40),w0=1, f=1)

#Propagate optical axis
plano_convex.propagate_ray(opt_axis)
biconcave.propagate_ray(opt_axis)
biconvex.propagate_ray(opt_axis)
pos_meniscus.propagate_ray(opt_axis)
plano_concave.propagate_ray(opt_axis)
screen.propagate_ray(opt_axis)

#Propagate paraxial ray 
plano_convex.propagate_ray(ray_p)
biconcave.propagate_ray(ray_p)
biconvex.propagate_ray(ray_p)
pos_meniscus.propagate_ray(ray_p)
plano_concave.propagate_ray(ray_p)
screen.propagate_ray(ray_p)

#Propgate marginal ray
plano_convex.propagate_ray(ray_m)
biconcave.propagate_ray(ray_m)
biconvex.propagate_ray(ray_m)
pos_meniscus.propagate_ray(ray_m)
plano_concave.propagate_ray(ray_m)
screen.propagate_ray(ray_m)

#Propgate best focus ray
plano_convex.propagate_ray(ray_bf)
biconcave.propagate_ray(ray_bf)
biconvex.propagate_ray(ray_bf)
pos_meniscus.propagate_ray(ray_bf)
plano_concave.propagate_ray(ray_bf)
screen.propagate_ray(ray_bf)

paraxial_focus_sys = ray_p.intersection(opt_axis)
marginal_focus_sys = ray_m.intersection(opt_axis)
best_focus_sys = ray_bf.intersection(opt_axis)

#Propagate bundle15
for ray_i in bundle15.spots():
    beam = Ray(ray_i, (0, 0, 1))
    plano_convex.propagate_ray(beam)
    biconcave.propagate_ray(beam)
    biconvex.propagate_ray(beam)
    pos_meniscus.propagate_ray(beam)
    plano_concave.propagate_ray(beam)
    screen.propagate_ray(beam)
    ax.plot(beam.coordinates()[0], 
            beam.coordinates()[2], 
            beam.coordinates()[1], 
            color = "mediumblue", linewidth = 0.3)
#Propagate bundle_angled_r
for ray_i in bundle_angled_r.spots():
    beam = Ray(ray_i, (0, 5, 45))
    plano_convex.propagate_ray(beam)
    biconcave.propagate_ray(beam)
    biconvex.propagate_ray(beam)
    pos_meniscus.propagate_ray(beam)
    plano_concave.propagate_ray(beam)
    screen.propagate_ray(beam)
    ax.plot(beam.coordinates()[0], 
            beam.coordinates()[2], 
            beam.coordinates()[1], 
            color = "red", linewidth = 0.3)


circle_sys_b = Circle((0, 0), bundle15._r+0.5, alpha = 0.1, color = "navy")
ax.add_patch(circle_sys_b)
art3d.pathpatch_2d_to_3d(circle_sys_b, z = 45, zdir = "y")

circle_sys_r = Circle((0, -5), bundle_angled_r._r+0.5, alpha = 0.1, color = 
                      "red")
ax.add_patch(circle_sys_r)
art3d.pathpatch_2d_to_3d(circle_sys_r, z = 45, zdir = "y")

x_patch = sp.array([[-20, 20]])
y_patch = sp.array([[min(screen.screen_spots()[2]),
                     min(screen.screen_spots()[2])]])
z_patch = sp.array([[-15, -15], [15, 15]])
ax.plot_surface(x_patch, y_patch, z_patch, alpha = 0.09, color = "k")

# plt.title("Schematic of a multi lens system", fontsize = 12)

ax.set_xlabel(r"$x$ $(mm)$", labelpad = 1, fontsize = 8)
ax.set_ylabel(r"$z$ $(mm)$", labelpad = 5.5)
ax.zaxis.set_rotate_label(False)
ax.set_zlabel(r"$y$ $(mm)$", labelpad = 1, rotation = 90)

# ax.xaxis.set_ticks([])
ax.tick_params(axis= "x", pad = 2, labelsize = 7)
plt.xticks(sp.arange(-20, 30, 10))

ax.tick_params(axis= "y", pad = 2, labelsize = 9)
ax.tick_params(axis = "z", pad = 3, labelsize = 9)



# plt.savefig("multi lens system.jpeg", dpi = 400)
plt.show()
    

#%%
from ray import Ray, Bundle
from optimizer import Optimization
# from ray_bundles import Bundle, Uniform
from opticalelements import SphericalRefraction, OutputPlane, Cavity
from matplotlib.patches import Circle, PathPatch
import scipy as sp
import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.transforms import Affine2D

# s = SphericalRefraction((0, 0, 100), (0, 0, 200), 
#                         curv_1 = 0.09, curv_2=-0.03, 
#                         n1 = 1, n2 = 1.5, 
#                         a_R = 1/0.03,
#                         Re_1 = 0.75,Re_2 = 1)
# s2 = SphericalRefraction((0, 0, 210), (0, 0, 310), 
#                         curv_1 = 0, curv_2=0, 
#                         n1 = 1, n2 = 1.5, 
#                         a_R = 1/0.0003,theta = 1*np.pi/180,
#                         Re_1 = 0.75,Re_2 = 1)

# raytest = Ray("Gaussian", 192300e9, (0,0,0), (0,0,1), A=1, phase=0,
              # w0 = 1e-3, f = 1e-9)
# raytest2 = Ray("Gaussian", 192300e9, (-2,0,0), (0,0,1), A=1, phase=0,
#               w0 = 1e-3, f = 1e-9)
# s2.propagate_ray(raytest,"refract", "refract")
# s2.propagate_ray(raytest2,"refract", "refract")
# s2.propagate_ray(raytest, action_2 = "refract", action_1 = "reflect")

# # s2.propagate_ray(raytest,"refract", None)
# # print(s2.n(raytest, s2.z_0_1))

# print(raytest.coordinates())
# # print(raytest.amplitudes())
# # print(raytest.E2s())
# # s.propagate_ray(raytest, "refract", None)


# op = OutputPlane((0,0,350))
# op.propagate_ray(raytest)
# op.propagate_ray(raytest2)
# print(raytest.coordinates()[2])
# print(raytest2.coordinates()[2])
# print(raytest.Es())
# print(raytest.amplitudes())

#%%


s2 = SphericalRefraction((0, 0, 210), (0, 0, 213), 
                        curv_1 = 0, curv_2=0, 
                        n1 = 1, n2 = 1.5, 
                        a_R = 1/0.0003,theta = 1*np.pi/180,
                        Re_1 = 0.75,Re_2 = 0.75)
s2copy = SphericalRefraction((0, 0, 210), (0, 0, 213), 
                        curv_1 = 0, curv_2=0, 
                        n1 = 1, n2 = 1.5, 
                        a_R = 1/0.0003,theta = 1*np.pi/180,
                        Re_1 = 0.75,Re_2 = 0.75)
raytest = Ray("Gaussian", np.array([192300e9, 192301e9]), (0,0,200), (0,0,1), A=1, phase=0,
w0 = 1e-3, f = 1e-15)
op = OutputPlane((0,0,250), "circle", R = 1)
b = Bundle(raytest)
rays = b.manyray(100)

c = Cavity(s2, s2copy, op, L = 5, theta =-0*np.pi/180)
c.resonate(rays)

# c = Cavity(s, s2, L = np.array([100, 200, 300]))


# c.test(raytest)

# op = OutputPlane((0,0,350))
# op.propagate_ray(raytest)

#%%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(azim = 30, elev = 23)

for ray in rays[0:]:
    ax.plot(ray.coordinates()[0], ray.coordinates()[1], ray.coordinates()[2], 
            label = "p: [0, 0.01], 0], k: [0, 0, 1]", linewidth = 1)
    # ax.plot(ray2.coordinates()[0], ray2.coordinates()[1], ray2.coordinates()[2], 
            # label = "p: [0, 0.01], 0], k: [0, 0, 1]", linewidth = 1)
    ax.set_xlabel("x")
    ax.set_zlabel("z")
    ax.set_ylabel("y")
    ax.set_xlim((0,5))

# ax.azim= 0
ax.elev = 0
#%%
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(azim = 30, elev = 23)


ax.plot(raytest.coordinates()[0], raytest.coordinates()[1], raytest.coordinates()[2], 
        label = "p: [0, 0.01], 0], k: [0, 0, 1]", linewidth = 1)
# ax.plot(raytest2.coordinates()[0], raytest2.coordinates()[1], raytest2.coordinates()[2], 
        # label = "p: [0, 0.01], 0], k: [0, 0, 1]", linewidth = 1)
ax.set_xlabel("x")
ax.set_zlabel("z")
ax.set_ylabel("y")
ax.set_xlim((-20,20))
ax.set_zlim((0, 400))
# ax.azim= 0
# ax.elev = 90

























