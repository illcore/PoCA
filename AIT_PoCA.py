import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
# Load data
data = pd.read_csv('Discrete_BESS_Pb.dat', sep=' ', header=None)

# Extract columns
ID = np.array(data[0])
xcor = data[1].to_numpy()
ycor = data[2].to_numpy()
zcor = data[3].to_numpy()
Energy = data[4].to_numpy()

# Initialize lists to store results
angles = []
poca_points = []

# Open file to write angles
with open("my_angles.txt", "w") as f:
    for x in range(len(ID) - 3):
        if ID[x] == ID[x + 3]:
            PrintID = ID[x]
            PrintEnergy = Energy[x]
            
            # Vectors for the hits
            vec1 = np.array([xcor[x + 1] - xcor[x], ycor[x + 1] - ycor[x], zcor[x + 1] - zcor[x]])
            vec2 = np.array([xcor[x + 3] - xcor[x + 2], ycor[x + 3] - ycor[x + 2], zcor[x + 3] - zcor[x + 2]])
            
            # Calculate magnitudes
            quotient1 = np.linalg.norm(vec1)
            quotient2 = np.linalg.norm(vec2)
            quotient = quotient1 * quotient2
            
            # Calculate scattering angle
            angle = math.acos(np.clip(np.dot(vec1, vec2) / quotient, -1.0, 1.0))
            angle = angle * 1000  # Convert to mrad
            
            # Calculate PoCA point
            A1, A2 = np.array([xcor[x], ycor[x], zcor[x]]), np.array([xcor[x + 1], ycor[x + 1], zcor[x + 1]])
            B1, B2 = np.array([xcor[x + 2], ycor[x + 2], zcor[x + 2]]), np.array([xcor[x + 3], ycor[x + 3], zcor[x + 3]])
            
            # PoCA calculation using vector parameterization
            def closest_point(A1, A2, B1, B2):
                u = A2 - A1
                v = B2 - B1
                w = A1 - B1
                a = np.dot(u, u)  # |u|^2
                b = np.dot(u, v)
                c = np.dot(v, v)  # |v|^2
                d = np.dot(u, w)
                e = np.dot(v, w)

                # Denominator of the system
                D = a * c - b * b
                
                if D < 1e-10:  # Lines are parallel
                    t = 0
                else:
                    t = (b * e - c * d) / D

                PoCA = A1 + t * u
                return PoCA
            
            poca = closest_point(A1, A2, B1, B2)
            poca_points.append(poca)
            angles.append(angle)
            
            # Write to file
            f.write(f"{PrintID} {PrintEnergy} {angle}\n")

# Convert PoCA points to a NumPy array for plotting
poca_points = np.array(poca_points)

plt.figure(figsize=(13,13), dpi=100)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['axes.linewidth'] = 1.5
plt.rc("font", size=25, family="Arial", weight='bold')
# Function to plot in XY or XZ plane or ZY plane
def plot_scattering_intensity(poca_points, angles, plane='XZ'):    
    if plane == 'XZ':
        sc = plt.scatter(poca_points[:, 0], poca_points[:, 2], c=angles, cmap='inferno', s=100)
        plt.xlabel(r'X-coordinate [cm]')
        plt.ylabel(r'Z-coordinate [cm]')
    elif plane == 'XY':
        sc = plt.scatter(poca_points[:, 0], poca_points[:, 1], c=angles, cmap='inferno', s=100)
        plt.xlabel(r'X-coordinate [cm]')
        plt.ylabel(r'Y-coordinate [cm]')
    elif plane == 'ZY':
        sc = plt.scatter(poca_points[:, 2], poca_points[:, 1], c=angles, cmap='inferno', s=100)
        plt.xlabel(r'Z-coordinate [cm]')
        plt.ylabel(r'Y-coordinate [cm]')    
    plt.colorbar(sc, label=r'Scattering angle [mrad]', orientation='horizontal')
    # Ticks
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', length=15, width=2,labelsize=22)
    plt.tick_params(axis='both', which='minor', length=7.5, width=1.5,labelsize=22)
    plt.xticks(np.arange(-50, 50.01, step=10))
    plt.yticks(np.arange(-50, 50.01, step=10))
    #Axes limits
    plt.xlim(-50, 50.01)
    plt.ylim(-50, 50.01)
    plt.savefig("PoCA_XZ_plane.pdf", bbox_inches='tight')
    plt.show()           
# Plotting in the XZ plane
plot_scattering_intensity(poca_points, angles, plane='XZ')

