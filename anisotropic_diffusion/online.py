import os
import sys
import time
import pickle
import numpy as np
import scipy.io as sio 
import pyamg
from scipy.sparse.linalg import spsolve, cg
from scipy.sparse import coo_matrix, hstack

try:
    from medium import compute_anisotropic_tensor, compute_local_anisotropic_tensor
    from form_system_2d import form_system_2d
    from form_source_2d import source_function, form_source_2d
    from apply_bc_2d import boundary_function, apply_bc_2d
    from bilinear_partition import bilinear_partition
    from form_global_basis import form_global_basis
    from solver import (
        save_SPLU,
        save_LDL,
        FOM_solution,
        ROM_solution,
    )
    from util import (
        get_args,
        reconstruct_field,
        plot_scalar_field,
        plot_scalar_field_with_contour,
        plot_surface,
        plot_tensor_components,
        plot_global_basis_with_coarse_grid,
    )
except ImportError as e:
    print(f"FATAL ERROR: Could not import a necessary module: {e}")
    print("Please ensure all translated Python files (.py) are in the same directory.")
    sys.exit(1)

params = get_args()

verbosity = params.verbosity
root_dir = params.root_dir

medium_type = params.medium_type
test_anisotropy_order = params.test_anisotropy_order

n_FOM_x = params.n_FOM_x
n_FOM_y = params.n_FOM_y

Nx = params.Nx
Ny = params.Ny

solve_FOM = params.solve_FOM
plot_solution_FOM = params.plot_solution_FOM

solve_ROM = params.solve_ROM
plot_solution_rom = params.plot_solution_rom
num_loc_basis = params.num_loc_basis

num_realizations = params.num_realizations
num_loc_bc_mode = params.num_loc_bc_mode
num_loc_snap = params.num_loc_snap
train_anisotropy_order = params.train_anisotropy_order

range_base_theta_pi = params.range_base_theta_pi
range_curv_x_pi = params.range_curv_x_pi
range_curv_y_pi = params.range_curv_y_pi
range_mag_grad_x = params.range_mag_grad_x
range_mag_grad_y = params.range_mag_grad_y

multiply_POU = params.multiply_POU

# Coordinates for the FOM grid (cell centers)
X1 = np.linspace(1/n_FOM_x/2, 1 - 1/n_FOM_x/2, n_FOM_x)
Y1 = np.linspace(1/n_FOM_y/2, 1 - 1/n_FOM_y/2, n_FOM_y)
X1_mesh, Y1_mesh = np.meshgrid(X1, Y1)

# Coordinates for the FOM grid (nodal points)
X_nodes = np.linspace(0, 1, n_FOM_x + 1)
Y_nodes = np.linspace(0, 1, n_FOM_y + 1)
X_nodes_mesh, Y_nodes_mesh = np.meshgrid(X_nodes, Y_nodes)

nx = n_FOM_x // Nx
ny = nx

np.random.seed(42)

prob_config = (
    f"{medium_type}_anis{test_anisotropy_order}_n{n_FOM_x}_N{Nx}"
)

samp_config = (
    f"anis{train_anisotropy_order}/nx{nx}_ny{ny}_real{num_realizations}_bc{num_loc_bc_mode}_snap{num_loc_snap}/"
    f"rbt{range_base_theta_pi:.2f}_rcx{range_curv_x_pi:.2f}_rcy{range_curv_y_pi:.2f}_rmgx{range_mag_grad_x:.2f}_rmgy{range_mag_grad_y:.2f}"
)

basis_dir = (
    f"{prob_config}/{samp_config}/POU{multiply_POU}_basis{num_loc_basis}"
)

run_dir = f"{basis_dir}/static" 
os.makedirs(run_dir, exist_ok=True)

try:
    Az, bx, by, a, contour_levels= compute_anisotropic_tensor(
        Ny=Ny*ny, 
        Nx=Nx*nx, 
        anisotropy_ratio=10.0**test_anisotropy_order, 
        problem=medium_type,
        save_dir=prob_config
    ) 
except:
    bx, by, a = compute_local_anisotropic_tensor(
                Ny=Ny*ny, 
                Nx=Nx*nx, 
                anisotropy_ratio=10.0**test_anisotropy_order, 
                base_theta=0.5*np.pi, 
                curv_x=0.25*np.pi, 
                curv_y=0.25*np.pi,
                log_base_mag=0.0, 
                mag_grad_x=0.0, 
                mag_grad_y=0.0,
                field_type="Global",
                save_dir=prob_config
    )
    Az = None
    contour_levels = None
if verbosity >= 2:
    print(f'a.shape = {a.shape}')

F_vals = source_function(X1_mesh, Y1_mesh, medium_type)

Global_DA, Global_M = form_system_2d(a, nx, ny, Nx, Ny)
Global_f = form_source_2d(F_vals, nx, ny, Nx, Ny)

is_DOF, DOF_idx_FOM, FOM_DA, FOM_M, FOM_RHS, bfc, dof_map, T = apply_bc_2d(
    medium_type, Global_DA, Global_M, Global_f, nx, ny, Nx, Ny
)

if solve_FOM == 0:
    FOM_data = np.load(f'{prob_config}/FOM_solution.npz')
    FOM_U = FOM_data['FOM_U']
    time_FOM = FOM_data['time_FOM']
    FOM_data.close() 
elif solve_FOM == 1:
    plot_tensor_components(a, f"{prob_config}/a.png")
    plot_scalar_field(F_vals, f"{prob_config}/FOM_f.png", title="Source function $f$")
    FOM_U, time_FOM = FOM_solution(FOM_DA, FOM_RHS, prob_config)
    np.savez(f'{prob_config}/FOM_solution.npz', FOM_U=FOM_U, time_FOM=time_FOM)

if solve_FOM >= 0:
    FOM_residual = np.linalg.norm(FOM_RHS - FOM_DA @ FOM_U)
    if verbosity >= 1:
        print(f"FOM solution residual = {FOM_residual:.2e}")
        print(f"CPU time for reference solution: {time_FOM} seconds")

    if plot_solution_FOM:
        if verbosity >= 1:
            print('Plotting Reference Solution...')

        plot_FOM = reconstruct_field(FOM_U, bfc, medium_type, dof_map, ny, Ny, nx, Nx)

        plot_scalar_field_with_contour(
            plot_FOM, 
            f"{prob_config}/FOM_sol.png",
            X1_mesh, Y1_mesh, 
            contour_data=Az, 
            contour_levels=contour_levels,
            title="Reference Solution $u_h$",
        )

        plot_surface(
            plot_FOM,
            X_nodes_mesh, Y_nodes_mesh,
            f"{prob_config}/FOM_sol_surf.png",
            title="Reference Solution $u_h$",
        )

if os.path.isfile(f'{basis_dir}/ROM_matrix.pkl'):
    with open(f'{basis_dir}/ROM_matrix.pkl', 'rb') as f:
        ROM_matrix_dict = pickle.load(f)
    glob_basis = ROM_matrix_dict['glob_basis']
    glob_basis_DOF = ROM_matrix_dict['glob_basis_DOF']
    ROM_DA = ROM_matrix_dict['ROM_DA']
    ROM_M = ROM_matrix_dict['ROM_M']
    print('Successfully loaded ROM matrix')
else:
    print('Computing ROM matrix')
    if multiply_POU == 1:
        POU_basis = bilinear_partition(nx, ny, Nx, Ny, medium_type)
    else:
        POU_basis = None

    if POU_basis is not None:
        POU_basis = POU_basis.tocsr().real
        if verbosity >= 2:
            print(f'Partition of unity shape = {POU_basis.shape}')

    samp_dir = f"{root_dir}/{samp_config}"
    glob_basis = form_global_basis(verbosity, samp_dir, nx, ny, Nx, Ny, bx, by, num_loc_basis, POU_basis, medium_type)
    if medium_type == "magnetic_island":
        glob_basis_DOF = T.T @ glob_basis
    else:
        glob_basis_DOF = glob_basis[DOF_idx_FOM, :]

    ROM_DA = glob_basis_DOF.T @ FOM_DA @ glob_basis_DOF
    ROM_DA = ROM_DA.real
    ROM_DA = 0.5 * (ROM_DA + ROM_DA.T)
    ROM_DA = ROM_DA.tocsc()
    ROM_M = glob_basis_DOF.T @ FOM_M @ glob_basis_DOF
    ROM_M = ROM_M.real
    ROM_M = 0.5 * (ROM_M + ROM_M.T)
    ROM_M = ROM_M.tocsc()

    with open(f'{basis_dir}/ROM_matrix.pkl', 'wb') as f:
        pickle.dump({
            'glob_basis': glob_basis,
            'glob_basis_DOF': glob_basis_DOF,
            'ROM_DA': ROM_DA,
            'ROM_M': ROM_M
        }, f)

if solve_ROM == 1 and not os.path.isfile(f'{basis_dir}/ROM_DA_LDL.npz'):
    print('Computing ROM sparse Cholesky factorization')
    save_LDL(ROM_DA, f"{basis_dir}/ROM_DA_LDL.npz")
if solve_ROM == 2 and not os.path.isfile(f'{basis_dir}/ROM_DA_SPLU.npz'):
    print('Computing ROM super LU factorization')
    save_SPLU(ROM_DA, f"{basis_dir}/ROM_DA_SPLU.npz")

if verbosity >= 2:
    print(f'Global basis shape = {glob_basis.shape}')
    print(f'Global basis DOF shape = {glob_basis_DOF.shape}')

if medium_type == "magnetic_island":
    subdomains = [0, Ny-1, (Nx-1)*(Ny)//2, (Nx-2)*(Ny), (Nx-1)*(Ny)-1]
else:
    subdomains = [0, Ny-2, ((Nx-2)*(Ny-1)+(Ny-2))//2, (Nx-2)*(Ny-1), (Nx-1)*(Ny-1)-1]
for subdomain_idx in subdomains:
    for basis_idx in range(min(num_loc_basis, 21)):
        plot_global_basis_with_coarse_grid(
            glob_basis, basis_idx, subdomain_idx, num_loc_basis, n_FOM_y+1, n_FOM_x+1, Ny, Nx, medium_type,  
            f"{run_dir}/Global_basis_{subdomain_idx}_{basis_idx}.png",
        )

ROM_RHS = glob_basis_DOF.T @ FOM_RHS

if solve_ROM == 0:
    ROM_data = np.load(f'{run_dir}/ROM_solution.npz')
    ROM_U = ROM_data['ROM_U']
    ROM_U_FOM = ROM_data['ROM_U_FOM']
    time_factor = ROM_data['time_factor']
    time_ROM = ROM_data['time_ROM']
    FOM_data.close() 
else:
    ROM_U, time_factor, time_ROM = ROM_solution(ROM_DA, ROM_RHS, basis_dir, solve_ROM)
    ROM_U_FOM = glob_basis_DOF @ ROM_U
    np.savez(f'{run_dir}/ROM_solution.npz', ROM_U=ROM_U, ROM_U_FOM=ROM_U_FOM, time_factor=time_factor, time_ROM=time_ROM)

ROM_residual = np.linalg.norm(ROM_RHS - ROM_DA @ ROM_U)
if verbosity >= 1:
    print(f"CPU time for ROM factorization: {time_factor} seconds")
    print(f"ROM solution residual = {ROM_residual:.2e}")
    print(f"CPU time for ROM solution: {time_ROM} seconds")

if plot_solution_rom:
    if verbosity >= 1:
        print('Plotting ROM Solution...')

    plot_ROM = reconstruct_field(ROM_U_FOM, bfc, medium_type, dof_map, ny, Ny, nx, Nx)

    plot_scalar_field_with_contour(
        plot_ROM, 
        f"{run_dir}/ROM_sol.png",
        X1_mesh, Y1_mesh, 
        contour_data=Az, 
        contour_levels=contour_levels,
        title="PU-DDFEM Solution $u_H$",
    )

    plot_surface(
        plot_ROM,
        X_nodes_mesh, Y_nodes_mesh,
        f"{run_dir}/ROM_sol_surf.png",
        title="Reference Solution $u_h$",
    )

if solve_FOM >= 0:
    if plot_solution_rom and plot_solution_FOM:
        plot_scalar_field(
            plot_FOM - plot_ROM,
            f"{run_dir}/ROM_err.png",
            title="Pointwise Error $u_h-u_H$",
        )

    error_diff = ROM_U_FOM - FOM_U
    
    error_sq_L2 = error_diff.T @ FOM_M @ error_diff
    FOM_sq_L2 = FOM_U.T @ FOM_M @ FOM_U
    L2_error = np.sqrt(error_sq_L2 / FOM_sq_L2)

    print(f"Speed-up: {time_FOM / time_ROM}")
    print(f'The relative L2 Error     is {L2_error:.2e}.')

print(f'Saved results to {run_dir}')
