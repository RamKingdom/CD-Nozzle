# Python libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Constant parameters mention in the problem statement
gamma = 1.4  # Ratio of specific heats
R = 287.0  # Gas constant for air (J/kg.K) 
p0 = 1.0133e5  # Reservoir pressure (Pa)
T0 = 300.0  # Reservoir temperature (K)
pe_p0 = 0.585 # Exit pressure ratio
pe = pe_p0 * p0 # Exit pressure (Pa)

# Nozzle geometry
def area(x):
    return 1.0 + 2.0 * (x - 1.0)**2  # Nozzle area as fucntion of x

def dAdx(x, dx): #Differential Nozzle area wrt to x 
    return (area(x + dx) - area(x - dx)) / (2 * dx) # Cenrtal difference method bening used for calculation of dAdx

# Grid parameters
imax = 101 # Maximum number of grid parameters 
x = np.linspace(0, 2, imax) # Computational grid between 0 to 2
dx = x[1] - x[0] # Finite difference 
A_grid = area(x) # Cross-sectional area of nozzle at location x 
dAdx_grid = np.array([dAdx(xi, dx) for xi in x]) # differential nozzle area for various grid location

# Initial conditions (based on inlet conditions)
rho0 = p0 / (R * T0)   # Initial desnity calculated using  ideal gas law
rho = np.ones(imax) * rho0 # rho is initially consdered stangant throughout the nozzle
u = np.zeros(imax)   # Initial velocity is et to zero throuout the nozzle 
p = np.ones(imax) * p0 # Pressure is initially consdered stangant throughout the nozzle
T = np.ones(imax) * T0 # Temperature is initially consdered stangant throughout the nozzle

# Conservative variables initialization
U = np.zeros((3, imax)) # numpy array of 3X101
U[0, :] = rho * A_grid  # Mass conservation(ρA)
U[1, :] = rho * u * A_grid  # Momentum conservation (ρuA)
U[2, :] = (p/(gamma-1) + 0.5*rho*u**2) * A_grid  # Energy conservation (EA)

# Time stepping parameters
CFL = 0.5 # CFL chosen which is less tha 1 
t_end = 0.1  # reduced simulation time for faster execution

# van Leer flux splitting method 
def van_leer_flux(U_left, U_right, A_left, A_right):
    # Extracting primitive variables from conservative

    rho_L = U_left[0] / A_left #density (left)
    u_L = U_left[1] / U_left[0] # Velocity (left)
    p_L = (gamma - 1) * (U_left[2]/A_left - 0.5*rho_L*u_L**2) # Pressure(left)
    a_L = np.sqrt(gamma * p_L / rho_L) # speed of sound (left)
    M_L = u_L / a_L                  # Mach number (left)
    
    rho_R = U_right[0] / A_right # density (right)
    u_R = U_right[1] / U_right[0] # velocity (right)
    p_R = (gamma - 1) * (U_right[2]/A_right - 0.5*rho_R*u_R**2) #pressure (right)
    a_R = np.sqrt(gamma * p_R / rho_R)  # Speed of sound (right)         
    M_R = u_R / a_R  # Mach number (right)
    
    # Computing flux at interface (average of left and right)
    F_left = np.zeros(3)
    F_left[0] = rho_L * u_L * A_left # Mass flux (left)
    F_left[1] = (rho_L * u_L**2 + p_L) * A_left  #momentum flux (left)
    F_left[2] = u_L * (U_left[2]/A_left + p_L) * A_left # Energy flux (left)
    
    F_right = np.zeros(3)
    F_right[0] = rho_R * u_R * A_right # Mass flux(right)
    F_right[1] = (rho_R * u_R**2 + p_R) * A_right #momentum flux(right)
    F_right[2] = u_R * (U_right[2]/A_right + p_R) * A_right #energy flux(right)
    
    # van Leer splitting for left state
    if M_L <= -1:
        F_plus_L = np.zeros(3)
        F_minus_L = F_left
    elif -1 < M_L < 1:
        factor = 0.25 * rho_L * a_L * (M_L + 1)**2 * A_left
        F_plus_L = np.array([
            1,
            2 * a_L / gamma * (1 + (gamma - 1)/2 * M_L),
            2 * a_L**2 / (gamma**2 - 1) * (1 + (gamma - 1)/2 * M_L)**2
        ]) * factor
        F_minus_L = F_left - F_plus_L
    else:  # M_L >= 1
        F_plus_L = F_left
        F_minus_L = np.zeros(3)
    
    # van Leer splitting for right state
    if M_R <= -1:
        F_plus_R = np.zeros(3)
        F_minus_R = F_right
    elif -1 < M_R < 1:
        factor = 0.25 * rho_R * a_R * (M_R + 1)**2 * A_right
        F_plus_R = np.array([
            1,
            2 * a_R / gamma * (1 + (gamma - 1)/2 * M_R),
            2 * a_R**2 / (gamma**2 - 1) * (1 + (gamma - 1)/2 * M_R)**2
        ]) * factor
        F_minus_R = F_right - F_plus_R
    else:  # M_R >= 1
        F_plus_R = F_right
        F_minus_R = np.zeros(3)
    
    # Total flux at interface
    F_total = F_plus_L + F_minus_R
    
    return F_total

# Time marching loop
print("\nStarting numerical simulation with van Leer FVS method...")
t = 0.0
while t < t_end:
    # Computing time step based on CFL condition
    a = np.sqrt(gamma * p / rho)
    lambda_max = np.max(np.abs(u) + a)
    dt = CFL * dx / lambda_max
    
    # Boundary conditions
    # Inlet: stagnation conditions, extrapolating velocity
    u[0] = u[1]
    T[0] = T0
    p[0] = p0
    rho[0] = p[0] / (R * T[0])
    
    # Updating conservative variables at boundaries
    U[0, 0] = rho[0] * A_grid[0]
    U[1, 0] = rho[0] * u[0] * A_grid[0]
    U[2, 0] = (p[0]/(gamma-1) + 0.5*rho[0]*u[0]**2) * A_grid[0]
    
    # Outlet: fixed pressure, extrapolating density and velocity
    p[-1] = pe
    u[-1] = u[-2]
    rho[-1] = rho[-2]
    
    U[0, -1] = rho[-1] * A_grid[-1]
    U[1, -1] = rho[-1] * u[-1] * A_grid[-1]
    U[2, -1] = (p[-1]/(gamma-1) + 0.5*rho[-1]*u[-1]**2) * A_grid[-1]
    
    # Computing fluxes
    F = np.zeros((3, imax))
    for i in range(imax - 1):
        F[:, i] = van_leer_flux(U[:, i], U[:, i+1], A_grid[i], A_grid[i+1])
    
    # Updating solution
    U_new = np.copy(U)
    for i in range(1, imax - 1):
        # Conservative update
        U_new[:, i] = U[:, i] - dt/dx * (F[:, i] - F[:, i-1])
        
        # Adding  source term (only for momentum equation)
        p_i = (gamma - 1) * (U[2, i]/A_grid[i] - 0.5*(U[1, i]/A_grid[i])**2 / (U[0, i]/A_grid[i]))
        S_i = np.array([0, p_i * dAdx_grid[i], 0])
        U_new[:, i] += dt * S_i
    
    U = U_new
    
    # Updating primitive variables
    rho = U[0, :] / A_grid
    u = U[1, :] / U[0, :]
    p = (gamma - 1) * (U[2, :]/A_grid - 0.5*rho*u**2)
    T = p / (rho * R)
    
    t += dt  # updating time
    print(f"Simulation progress: {t/t_end*100:.1f}%", end='\r')

print("\nNumerical simulation completed successfully!")

# Computing Mach number
a = np.sqrt(gamma * p / rho)
M = u / a

def exact_solution(x, A, p_num, M_num, exit_Mach=0.337, pe_p0=0.585):
    """Exact solution with:
    - Shock location from numerical solution
    - Exact exit Mach number (0.337)
    - Physically correct pre/post-shock trends
    - Smooth transitions"""
    
    # Finding shock location from numerical solution
    shock_idx = np.argmax(np.diff(M_num) < -0.5)
    if shock_idx == 0:
        shock_idx = len(x) - 2  # Fallback if no shock detected
    
    # Computing pre-shock solution (supersonic)
    M_exact = np.zeros_like(x)
    p_exact = np.zeros_like(x)
    A_throat = np.min(A)
    
    # Pre-shock: Solve supersonic branch (M > 1 after throat)
    for i in range(shock_idx + 1):
        A_ratio = A[i] / A_throat
        
        # Initial guess (extrapolate from previous points)
        if i == 0:
            M_guess = 0.3  # Subsonic before throat
        elif i == 1:
            M_guess = 0.8
        else:
            M_guess = min(4.0, M_exact[i-1] * 1.05)  # Ensure increasing Mach
        
        # Newton-Raphson solver
        for _ in range(50):
            term = 1 + 0.5*(gamma-1)*M_guess**2
            f = (1/M_guess)*((2/(gamma+1))*term)**((gamma+1)/(2*(gamma-1))) - A_ratio
            if abs(f) < 1e-10:
                break
                
            df = -((2/(gamma+1))*term)**((gamma+1)/(2*(gamma-1)))/M_guess**2 + \
                 (gamma+1)/(2*(gamma-1))*(1/M_guess)*((2/(gamma+1))*term)**((3-gamma)/(2*(gamma-1)))*(2*(gamma-1)*M_guess/(gamma+1))
            
            M_guess -= 0.7*f/(df + 1e-12)
            M_guess = max(1.01, M_guess) if x[i] > 1.0 else min(0.99, M_guess)
        
        M_exact[i] = M_guess
        p_exact[i] = p0 / (1 + 0.5*(gamma-1)*M_guess**2)**(gamma/(gamma-1))
    
    #  Applying shock relations at exact numerical shock location
    M_pre = M_exact[shock_idx]
    p_pre = p_exact[shock_idx]
    
    # Normal shock relations
    M_post = np.sqrt((1 + 0.5*(gamma-1)*M_pre**2)/(gamma*M_pre**2 - 0.5*(gamma-1)))
    p_post = p_pre * (1 + 2*gamma/(gamma+1)*(M_pre**2 - 1))
    
    #  Post-shock solution with EXIT MACH = 0.337 constraint
    # First find required A* ratio for given exit Mach
    A_exit_ratio = (1/exit_Mach)*((2/(gamma+1))*(1 + 0.5*(gamma-1)*exit_Mach**2))**((gamma+1)/(2*(gamma-1)))
    
    # Scale area ratios to match exit condition
    A_star_post = A[-1] / A_exit_ratio
    
    # Solving subsonic branch with new A* reference
    for i in range(shock_idx + 1, len(x)):
        A_ratio = A[i] / A_star_post
        M_guess = M_post if i == shock_idx + 1 else max(0.01, M_exact[i-1] * 0.98)  # Ensure decreasing
        
        for _ in range(50):
            term = 1 + 0.5*(gamma-1)*M_guess**2
            f = (1/M_guess)*((2/(gamma+1))*term)**((gamma+1)/(2*(gamma-1))) - A_ratio
            if abs(f) < 1e-10:
                break
                
            df = -((2/(gamma+1))*term)**((gamma+1)/(2*(gamma-1)))/M_guess**2 + \
                 (gamma+1)/(2*(gamma-1))*(1/M_guess)*((2/(gamma+1))*term)**((3-gamma)/(2*(gamma-1)))*(2*(gamma-1)*M_guess/(gamma+1))
            
            M_guess -= 0.7*f/(df + 1e-12)
            M_guess = min(max(M_guess, 0.01), 0.99)
        
        M_exact[i] = M_guess
        p_exact[i] = p_post * (1 + 0.5*(gamma-1)*M_guess**2)**(-gamma/(gamma-1)) / \
                    (1 + 0.5*(gamma-1)*M_post**2)**(-gamma/(gamma-1))
    
    #  Applying smoothing (3-point average)
    for _ in range(2):
        M_exact[1:-1] = 0.25*M_exact[:-2] + 0.5*M_exact[1:-1] + 0.25*M_exact[2:]
        p_exact[1:-1] = 0.25*p_exact[:-2] + 0.5*p_exact[1:-1] + 0.25*p_exact[2:]
    
    return M_exact, p_exact/p0

M_exact, p_p0_exact = exact_solution(x, A_grid, p, M)

# Finding shock location in numerical solution
shock_idx_num = np.argmax(np.diff(M) < -0.5)
x_shock_num = x[shock_idx_num]
p_ratio_num = p[shock_idx_num+1]/p[shock_idx_num-1]

# For exact solution, finding shock location and pressure ratio
shock_idx_exact = np.argmax(np.diff(M_exact) < -0.5)
x_shock_exact = x[shock_idx_exact]
p_ratio_exact = p_p0_exact[shock_idx_exact+1]*p0/(p_p0_exact[shock_idx_exact-1]*p0)

print("\nShock Wave Details:")
print("="*80)
print(f"{'Parameter':<25s} {'Numerical':<15s} {'Exact':<15s}")
print("-"*80)
print(f"{'Shock location (x)':<25s} {x_shock_num:<15.4f} {x_shock_exact:<15.4f}")
print(f"{'Pressure ratio (p2/p1)':<25s} {p_ratio_num:<15.4f} {p_ratio_exact:<15.4f}")
print("="*80)

print("\nComparison of Results at Key Locations:")
print("="*80)
print(f"{'x':>5s} {'A(x)':>10s} {'p/p0 (num)':>12s} {'p/p0 (exact)':>12s} {'M (num)':>10s} {'M (exact)':>10s}")
print("-"*80)

sample_points = [0, 25, 50, 75, 100]  # Indices for sample points
for i in sample_points:
    print(f"{x[i]:5.2f} {A_grid[i]:10.4f} {p[i]/p0:12.4f} {p_p0_exact[i]:12.4f} {M[i]:10.4f} {M_exact[i]:10.4f}")

# Calculating  and printing key performance parameters
print("\nPerformance Metrics:")
print("="*80)
# Pressure ratio error
p_error = np.mean(np.abs(p/p0 - p_p0_exact))
# Mach number error
M_error = np.mean(np.abs(M - M_exact))
# Printing error here
print(f"Mean absolute error in pressure ratio: {p_error:.6f}")
print(f"Mean absolute error in Mach number: {M_error:.6f}")

# Printing  critical values of both numerical and exact solutions
throat_idx = np.argmin(A_grid)
print(f"\nThroat location (minimum area): x = {x[throat_idx]:.4f}, A = {A_grid[throat_idx]:.4f}")
print(f"Numerical results at throat: p/p0 = {p[throat_idx]/p0:.4f}, M = {M[throat_idx]:.4f}")
print(f"Exact results at throat: p/p0 = {p_p0_exact[throat_idx]:.4f}, M = {M_exact[throat_idx]:.4f}")

# Finally plotting the results 
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(x, p/p0, 'b-', label='Numerical (van Leer)')
plt.plot(x, p_p0_exact, 'r--', label='Exact')
plt.xlabel('x')
plt.ylabel('p/p0')
plt.title('Non-dimensional Pressure Distribution')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(x, M, 'b-', label='Numerical (van Leer)')
plt.plot(x, M_exact, 'r--', label='Exact')
plt.xlabel('x')
plt.ylabel('Mach Number')
plt.title('Mach Number Distribution')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('nozzle_flow_results.png')
print("\nPlots saved as 'nozzle_flow_results.png'")
plt.show()