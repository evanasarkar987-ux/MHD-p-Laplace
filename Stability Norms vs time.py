from mpi4py import MPI
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem


def solve_problem(n, dt_input, T):
    domain = mesh.create_unit_cube(
        MPI.COMM_WORLD,
        n,
        n,
        n,
        cell_type=mesh.CellType.tetrahedron,
    )

    gdim = domain.geometry.dim
    cell = domain.basix_cell()

    B_element = element("Nedelec 1st kind H(curl)", cell, 1)
    u_element = element("Lagrange", cell, 2, shape=(gdim,))
    p_element = element("Lagrange", cell, 1)

    W = fem.functionspace(domain, mixed_element([B_element, u_element, p_element]))

    w = fem.Function(W)
    w_old = fem.Function(W)

    B, u, p = ufl.split(w)
    B_old, u_old, p_old = ufl.split(w_old)

    C, v, q = ufl.TestFunctions(W)

    nu = fem.Constant(domain, 1.0)
    dt = fem.Constant(domain, dt_input)
    time = fem.Constant(domain, 0.0)

    p_value = 10
    eps = fem.Constant(domain, 1.0e-8)

    x = ufl.SpatialCoordinate(domain)
    zero = fem.Constant(domain, 0.0)
    exp_t = ufl.exp(-time)

    u_exact = ufl.as_vector((
        exp_t * ufl.sin(np.pi * x[1]),
        exp_t * ufl.sin(np.pi * x[2]),
        exp_t * ufl.sin(np.pi * x[0]),
    ))

    B_exact = ufl.as_vector((
        exp_t * ufl.sin(np.pi * x[0]) * ufl.cos(np.pi * x[1]) * ufl.cos(np.pi * x[2]),
        -exp_t * ufl.cos(np.pi * x[0]) * ufl.sin(np.pi * x[1]) * ufl.cos(np.pi * x[2]),
        zero,
    ))

    p_exact = (
        exp_t
        * ufl.sin(np.pi * x[0])
        * ufl.sin(np.pi * x[1])
        * ufl.sin(np.pi * x[2])
    )

    u_t_exact = ufl.as_vector((
        -exp_t * ufl.sin(np.pi * x[1]),
        -exp_t * ufl.sin(np.pi * x[2]),
        -exp_t * ufl.sin(np.pi * x[0]),
    ))

    B_t_exact = ufl.as_vector((
        -exp_t * ufl.sin(np.pi * x[0]) * ufl.cos(np.pi * x[1]) * ufl.cos(np.pi * x[2]),
        exp_t * ufl.cos(np.pi * x[0]) * ufl.sin(np.pi * x[1]) * ufl.cos(np.pi * x[2]),
        zero,
    ))

    grad_u = ufl.grad(u)
    grad_u_norm = ufl.sqrt(ufl.inner(grad_u, grad_u) + eps)

    grad_exact = ufl.grad(u_exact)
    grad_exact_norm = ufl.sqrt(ufl.inner(grad_exact, grad_exact) + eps)

    F_current = (
        ufl.inner((B - B_old) / dt, C) * ufl.dx
        + ufl.inner(ufl.curl(B), ufl.curl(C)) * ufl.dx
        - ufl.inner(ufl.cross(u, B), ufl.curl(C)) * ufl.dx

        + ufl.inner((u - u_old) / dt, v) * ufl.dx
        + ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.inner(ufl.cross(B, ufl.curl(B)), v) * ufl.dx
        + nu * grad_u_norm**(p_value - 2.0) * ufl.inner(grad_u, ufl.grad(v)) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )

    F_exact = (
        ufl.inner(B_t_exact, C) * ufl.dx
        + ufl.inner(ufl.curl(B_exact), ufl.curl(C)) * ufl.dx
        - ufl.inner(ufl.cross(u_exact, B_exact), ufl.curl(C)) * ufl.dx

        + ufl.inner(u_t_exact, v) * ufl.dx
        + ufl.inner(ufl.dot(ufl.grad(u_exact), u_exact), v) * ufl.dx
        - p_exact * ufl.div(v) * ufl.dx
        + ufl.inner(ufl.cross(B_exact, ufl.curl(B_exact)), v) * ufl.dx
        + nu * grad_exact_norm**(p_value - 2.0) * ufl.inner(grad_exact, ufl.grad(v)) * ufl.dx
        + q * ufl.div(u_exact) * ufl.dx
    )

    residual = F_current - F_exact

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain,
        fdim,
        lambda x: np.full(x.shape[1], True, dtype=bool),
    )

    B_space, _ = W.sub(0).collapse()
    u_space, _ = W.sub(1).collapse()
    p_space, _ = W.sub(2).collapse()

    B_boundary = fem.Function(B_space)
    u_boundary = fem.Function(u_space)

    p_zero = fem.Function(p_space)
    p_zero.x.array[:] = 0.0

    def B_numpy(t):
        exp_t = np.exp(-t)
        return lambda x: np.vstack((
            exp_t * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]) * np.cos(np.pi * x[2]),
            -exp_t * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.cos(np.pi * x[2]),
            np.zeros_like(x[0]),
        ))

    def u_numpy(t):
        exp_t = np.exp(-t)
        return lambda x: np.vstack((
            exp_t * np.sin(np.pi * x[1]),
            exp_t * np.sin(np.pi * x[2]),
            exp_t * np.sin(np.pi * x[0]),
        ))

    B_dofs = fem.locate_dofs_topological(
        (W.sub(0), B_space),
        fdim,
        boundary_facets,
    )

    u_dofs = fem.locate_dofs_topological(
        (W.sub(1), u_space),
        fdim,
        boundary_facets,
    )

    pressure_dofs = fem.locate_dofs_geometrical(
        (W.sub(2), p_space),
        lambda x: np.logical_and.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[2], 0.0),
        )),
    )

    bc_B = fem.dirichletbc(B_boundary, B_dofs, W.sub(0))
    bc_u = fem.dirichletbc(u_boundary, u_dofs, W.sub(1))
    bc_p = fem.dirichletbc(p_zero, pressure_dofs, W.sub(2))

    problem = NonlinearProblem(
        residual,
        w,
        bcs=[bc_B, bc_u, bc_p],
        petsc_options_prefix="mhd_stability_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-8,
            "snes_max_it": 50,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_error_if_not_converged": True,
            "ksp_error_if_not_converged": True,
        },
    )

    time.value = 0.0
    w_old.sub(0).interpolate(B_numpy(0.0))
    w_old.sub(1).interpolate(u_numpy(0.0))
    w_old.sub(2).interpolate(lambda x: np.zeros_like(x[0]))
    w.x.array[:] = w_old.x.array[:]

    n_steps = int(round(T / dt_input))
    dt.value = T / n_steps

    dx_norm = ufl.dx(metadata={"quadrature_degree": 10})

    print("")
    print("Stability norms")
    print("t          ||B||_L2          ||curl B||_L2     ||u||_L2          ||grad u||_L2      ||u||_Linf" )

    for step in range(n_steps):
        current_time = (step + 1) * float(dt.value)
        time.value = current_time

        B_boundary.interpolate(B_numpy(current_time))
        u_boundary.interpolate(u_numpy(current_time))

        w.sub(0).interpolate(B_numpy(current_time))
        w.sub(1).interpolate(u_numpy(current_time))
        w.sub(2).interpolate(lambda x: np.zeros_like(x[0]))

        problem.solve()

        Bh, uh, ph = w.split()
        u_Linf = np.max(np.abs(uh.x.array))        
        B_L2_form = fem.form(ufl.inner(Bh, Bh) * dx_norm)
        curl_B_L2_form = fem.form(ufl.inner(ufl.curl(Bh), ufl.curl(Bh)) * dx_norm)

        u_L2_form = fem.form(ufl.inner(uh, uh) * dx_norm)
        grad_u_L2_form = fem.form(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * dx_norm)
        grad_uh = ufl.grad(uh)
        

        B_L2_norm = np.sqrt(domain.comm.allreduce(
            fem.assemble_scalar(B_L2_form),
            op=MPI.SUM
        ))

        curl_B_L2_norm = np.sqrt(domain.comm.allreduce(
            fem.assemble_scalar(curl_B_L2_form),
            op=MPI.SUM
        ))

        u_L2_norm = np.sqrt(domain.comm.allreduce(
            fem.assemble_scalar(u_L2_form),
            op=MPI.SUM
        ))


        grad_u_L2_norm = np.sqrt(domain.comm.allreduce(
            fem.assemble_scalar(grad_u_L2_form),
            op=MPI.SUM
        ))
        
        u_Linf_norm = np.max(np.abs(uh.x.array))
        print(
            f"{current_time:<10.4f} "
            f"{B_L2_norm:<18.6e} "
            f"{curl_B_L2_norm:<18.6e} "
            f"{u_L2_norm:<18.6e} "
            f"{grad_u_L2_norm:<18.6e}"
            f"{u_Linf:<17.6e}"
        )

        w_old.x.array[:] = w.x.array[:]


n = 4
dt = 0.05
T = 1.0

print(f"Running stability check with n={n}, dt={dt}, T={T}")
solve_problem(n, dt, T)
