import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def compute_anisotropic_tensor(
    Ny: int, 
    Nx: int, 
    anisotropy_ratio: float, 
    problem: str,
    save_dir: str = "."
) -> np.ndarray:
    """
    Computes the 2D anisotropic conductivity tensor K for specific prototype 
    problems from the Vogl et al. (2023) paper.
    
    Returns:
        np.ndarray: Shape (Ny, Nx, 2, 2) where K[i, j] is the 2x2 tensor.
    """
    kappa_par = 1.0
    kappa_perp = kappa_par / anisotropy_ratio
    diff = kappa_par - kappa_perp
    x0, y0 = 0.5, 0.5
    
    x_coords = np.linspace(1/Nx/2, 1 - 1/Nx/2, Nx)
    y_coords = np.linspace(1/Ny/2, 1 - 1/Ny/2, Ny)
    if problem == "constant_field":
        x_coords = x_coords * np.pi
        
    X, Y = np.meshgrid(x_coords, y_coords)
    
    if problem == "constant_field":
        Az = Y
        levels = np.linspace(0.0, 1.0, 11)
    elif problem == "single_null":
        d1_sq = (X - 0.5)**2 + (Y - 0.75)**2
        d2_sq = (X - 0.5)**2 + (Y + 0.25)**2
        Az = 0.5 * np.log(d1_sq * d2_sq + 1e-9)
        levels = np.linspace(-3.6, -0.6, 11)
    elif problem == "double_null":
        Az = 0.5*(X - x0)**2 + 0.5*(0.25 * np.sin(2*np.pi*(Y - y0)))**2
        levels = np.linspace(0.01, 0.14, 14)
    elif problem == "magnetic_island":
        L = 1.0
        envelope = (1.0 - ((X - x0)/(L/2))**2)
        Az = 0.5*(X - x0)**2 + 0.5*(envelope * 0.25 * np.sin(2*np.pi*(Y - y0)))**2
        levels = np.linspace(0.01, 0.12, 12)
    else:
        raise ValueError(f"Unknown problem type: {problem}")

    dAz_dy, dAz_dx = np.gradient(Az, y_coords, x_coords)

    Bpx, Bpy = dAz_dy, -dAz_dx
    
    B_mag = np.sqrt(Bpx**2 + Bpy**2 + 1.0)
    bx, by = Bpx/B_mag, Bpy/B_mag
    
    K = np.zeros((Ny, Nx, 2, 2))
    
    K[:, :, 0, 0] = kappa_perp + diff * bx**2
    K[:, :, 1, 1] = kappa_perp + diff * by**2
    K[:, :, 0, 1] = diff * bx * by
    K[:, :, 1, 0] = K[:, :, 0, 1] 
    
    def clip_vectors(vx, vy, percentile=95):
        mag = np.sqrt(vx**2 + vy**2)
        thresh = np.percentile(mag, percentile)
        # Avoid division by zero
        scale = np.ones_like(mag)
        mask = mag > thresh
        scale[mask] = thresh / mag[mask]
        return vx * scale, vy * scale

    plt.figure()
    im0 = plt.contourf(X, Y, Az, cmap="magma")
    cbar = plt.colorbar(im0)
    cbar.ax.tick_params(labelsize=16)
    im_lines = plt.contour(X, Y, Az, levels=levels, colors="w", linewidths=0.8)
    plt.clabel(im_lines, inline=True, fontsize=8, fmt="%.2f")
    plt.title("Magnetic flux $A_z$", fontsize=16)

    plt.gca().set_aspect('equal')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.xticks([0, 0.25, 0.5, 0.75, 1])
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/magnetic_flux_Az.png")
    plt.close()

    plt.figure()
    skip = Nx // 8 - 1
    bx_c, by_c = clip_vectors(bx, by)
    plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], bx_c[::skip, ::skip], by_c[::skip, ::skip], color='blue')
    plt.title("Effective directional vector $\hat{b}$", fontsize=16)

    plt.gca().set_aspect('equal')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.xticks([0, 0.25, 0.5, 0.75, 1])
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/effective_vector_b.png")
    plt.close()

    plt.figure()
    data = K[:, :, 0, 0]
    vmin = data[data > 0].min() if (data > 0).any() else None
    vmax = data.max()
    im3 = plt.pcolormesh(X, Y, K[:, :, 0, 0], shading='auto', cmap='magma', norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.title("Diffusion tensor component $K_{11}$", fontsize=16)
    cbar = plt.colorbar(im3)
    cbar.ax.tick_params(labelsize=16)

    plt.gca().set_aspect('equal')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.xticks([0, 0.25, 0.5, 0.75, 1])
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/diffusion_tensor_K11.png")
    plt.close()

    plt.figure()
    data = K[:, :, 1, 1]
    vmin = data[data > 0].min() if (data > 0).any() else None
    vmax = data.max()
    im4 = plt.pcolormesh(X, Y, K[:, :, 1, 1], shading='auto', cmap='magma', norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.title("Diffusion tensor component $K_{22}$", fontsize=16)
    cbar = plt.colorbar(im4)
    cbar.ax.tick_params(labelsize=16)

    plt.gca().set_aspect('equal')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.xticks([0, 0.25, 0.5, 0.75, 1])
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/diffusion_tensor_K22.png")
    plt.close()

    plt.figure()
    data = K[:, :, 0, 1]
    vmin = data[data > 0].min() if (data > 0).any() else None
    vmax = data.max()
    im5 = plt.pcolormesh(X, Y, K[:, :, 0, 1], shading='auto', cmap='magma', norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.title("Diffusion tensor component $K_{12}$", fontsize=16)
    cbar = plt.colorbar(im5)
    cbar.ax.tick_params(labelsize=16)

    plt.gca().set_aspect('equal')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.xticks([0, 0.25, 0.5, 0.75, 1])
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/diffusion_tensor_K12.png")
    plt.close()

    return Az, bx, by, K, levels

def compute_local_anisotropic_tensor(
    Ny: int, 
    Nx: int, 
    anisotropy_ratio: float,
    base_theta_pi: float,
    curv_x_pi: float,
    curv_y_pi: float,
    mag_grad_x: float,
    mag_grad_y: float,
    field_type: str = "",
    save_dir: str = "."
) -> np.ndarray:
    """
    Computes a 2D anisotropic conductivity tensor K using a local log-linear 
    parameterization of magnetic field angle and magnitude.

    The magnetic field is modeled as B = (Bp*cos(theta), Bp*sin(theta), 1.0),
    where the poloidal magnitude Bp follows an exponential profile to ensure 
    positivity and mimic physical field gradients (e.g., near null points).

    Args:
        Ny, Nx (int): Grid resolution.
        anisotropy_ratio (float): Ratio of parallel (kappa_par) to 
            perpendicular (kappa_perp) conductivity.
        base_theta_pi (float): The poloidal angle theta (pi radians) at the 
            origin (center of the patch).
        curv_x_pi, curv_y_pi (float): Spatial derivatives of the angle (d_theta/dx, 
            d_theta/dy), representing field curvature or shear.
        mag_grad_x, mag_grad_y (float): Spatial derivatives of the log-magnitude 
            (d(ln Bp)/dx, d(ln Bp)/dy), representing field strength gradients.

    Returns:
        np.ndarray: Shape (Ny, Nx, 2, 2) where K[i, j] is the 2x2 local 
            anisotropic conductivity tensor.
    """
    kappa_par = 1.0
    kappa_perp = kappa_par / anisotropy_ratio
    diff = kappa_par - kappa_perp
    
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)
    
    Bp = 1 / (1 + np.exp(mag_grad_x * X + mag_grad_y * Y))
    theta = (base_theta_pi + (curv_x_pi * X) + (curv_y_pi * Y)) * np.pi
    bx = Bp * np.cos(theta)
    by = Bp * np.sin(theta)
    
    K = np.zeros((Ny, Nx, 2, 2))
    
    K[:, :, 0, 0] = kappa_perp + diff * bx**2
    K[:, :, 1, 1] = kappa_perp + diff * by**2
    K[:, :, 0, 1] = diff * bx * by
    K[:, :, 1, 0] = K[:, :, 0, 1]
    
    if not field_type == "":
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(fr"{field_type} Parameter: $\varphi_0$={base_theta_pi:.2f} pi, $\alpha$={curv_x_pi:.2f} pi, $\beta$={curv_y_pi:.2f} pi, $\gamma$={mag_grad_x:.2f}, $\delta$={mag_grad_y:.2f}")

        def clip_vectors(vx, vy, percentile=95):
            mag = np.sqrt(vx**2 + vy**2)
            thresh = np.percentile(mag, percentile)
            # Avoid division by zero
            scale = np.ones_like(mag)
            mask = mag > thresh
            scale[mask] = thresh / mag[mask]
            return vx * scale, vy * scale

        def get_vrange(data):
            dmin, dmax = np.min(data), np.max(data)
            if np.isclose(dmin, dmax, atol=1e-12):
                return dmin - 0.1, dmax + 0.1
            return dmin, dmax

        plt.figure()
        skip = Nx // 8 - 1
        bx_c, by_c = clip_vectors(bx, by)
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], bx_c[::skip, ::skip], by_c[::skip, ::skip], color='blue')
        plt.title("Effective directional vector $\hat{b}$", fontsize=16)

        plt.gca().set_aspect('equal')
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.xticks([0, 0.25, 0.5, 0.75, 1])
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{field_type}_parametric_{base_theta_pi:.2f}_{curv_x_pi:.2f}_{curv_y_pi:.2f}_{mag_grad_x:.2f}_{mag_grad_y:.2f}_effective_vector_b.png")
        plt.close()

        plt.figure()
        vmin, vmax = get_vrange(K[:, :, 0, 0])
        im3 = plt.pcolormesh(X, Y, K[:, :, 0, 0], shading='auto', cmap='magma', vmin=vmin, vmax=vmax)
        plt.title("Diffusion tensor component $K_{11}$", fontsize=16)
        cbar = plt.colorbar(im3)
        cbar.ax.tick_params(labelsize=16)

        plt.gca().set_aspect('equal')
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.xticks([0, 0.25, 0.5, 0.75, 1])
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{field_type}_parametric_{base_theta_pi:.2f}_{curv_x_pi:.2f}_{curv_y_pi:.2f}_{mag_grad_x:.2f}_{mag_grad_y:.2f}_diffusion_tensor_K11.png")
        plt.close()

        plt.figure()
        vmin, vmax = get_vrange(K[:, :, 1, 1])
        im4 = plt.pcolormesh(X, Y, K[:, :, 1, 1], shading='auto', cmap='magma', vmin=vmin, vmax=vmax)
        plt.title("Diffusion tensor component $K_{12}$", fontsize=16)
        cbar = plt.colorbar(im4)
        cbar.ax.tick_params(labelsize=16)

        plt.gca().set_aspect('equal')
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.xticks([0, 0.25, 0.5, 0.75, 1])
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{field_type}_parametric_{base_theta_pi:.2f}_{curv_x_pi:.2f}_{curv_y_pi:.2f}_{mag_grad_x:.2f}_{mag_grad_y:.2f}_diffusion_tensor_K12.png")
        plt.close()

        plt.figure()
        v_xy = max(abs(np.min(K[:, :, 0, 1])), abs(np.max(K[:, :, 0, 1])))
        if v_xy < 1e-12: 
            v_xy = 0.1
        im5 = plt.pcolormesh(X, Y, K[:, :, 0, 1], shading='auto', cmap='magma', vmin=-v_xy, vmax=v_xy)
        plt.title("Diffusion tensor component $K_{22}$", fontsize=16)
        cbar = plt.colorbar(im5)
        cbar.ax.tick_params(labelsize=16)

        plt.gca().set_aspect('equal')
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.xticks([0, 0.25, 0.5, 0.75, 1])
        plt.yticks([0, 0.25, 0.5, 0.75, 1])
        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{field_type}_parametric_{base_theta_pi:.2f}_{curv_x_pi:.2f}_{curv_y_pi:.2f}_{mag_grad_x:.2f}_{mag_grad_y:.2f}_diffusion_tensor_K22.png")
        plt.close()

        return bx, by, K
