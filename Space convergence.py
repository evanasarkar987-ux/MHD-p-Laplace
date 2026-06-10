from mpi4py import MPI
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem


def solve_problem(n, dt_input, T, h_value):
    domain = mesh.create_unit_cube(
        MPI.COMM_WORLD,
        n,
        n,
        n,
        cell_type=mesh.CellType.tetrahedron,
    )

    gdim = domain.geometry.dim
    cell = domain.basix_cell()

    B_element = element("Nedelec 1st kind H(curl)", cell, 2)
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
    p_value = 5
    eps = fem.Constant(domain, 1.0e-8)

    x = ufl.SpatialCoordinate(domain)
    zero = fem.Constant(domain,0.0)
    # Exact solution:
    # u = (sin(y) sin(t), 0, 0)
    # p = 0
    # B = (0, sin(x) sin(t), 0)
    u_exact = ufl.as_vector((
        ufl.sin(x[1]) * ufl.sin(time),
        zero,
        zero,
    ))

    B_exact = ufl.as_vector((
        zero,
        ufl.sin(x[0]) * ufl.sin(time),
        zero,
    ))

    p_exact = zero

    u_t_exact = ufl.as_vector((
        ufl.sin(x[1]) * ufl.cos(time),
        zero,
        zero,
    ))

    B_t_exact = ufl.as_vector((
        zero,
        ufl.sin(x[0]) * ufl.cos(time),
        zero,
    ))
    grad_exact = ufl.grad(u_exact)
    grad_exact_norm = ufl.sqrt(ufl.inner(grad_exact, grad_exact) + eps)

    

    grad_u = ufl.grad(u)
    grad_u_norm = ufl.sqrt(ufl.inner(grad_u, grad_u) + eps)
    F_current = (
        ufl.inner((B - B_old) / dt, C) * ufl.dx
        + ufl.inner(ufl.curl(B), ufl.curl(C)) * ufl.dx
        - ufl.inner(ufl.cross(u, B), ufl.curl(C)) * ufl.dx

        + ufl.inner((u - u_old) / dt, v) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
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
        + ufl.inner(ufl.grad(u_exact) * u_exact, v) * ufl.dx
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
        return lambda x: np.vstack((
            np.zeros_like(x[0]),
            np.sin(x[0]) * np.sin(t),
            np.zeros_like(x[0]),
        ))

    def u_numpy(t):
        return lambda x: np.vstack((
            np.sin(x[1]) * np.sin(t),
            np.zeros_like(x[0]),
            np.zeros_like(x[0]),
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
        petsc_options_prefix=f"mhd_dth_{n}_",
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

    # Initial condition at t = 0
    time.value = 0.0
    w_old.sub(0).interpolate(B_numpy(0.0))
    w_old.sub(1).interpolate(u_numpy(0.0))
    w_old.sub(2).interpolate(lambda x: np.zeros_like(x[0]))
    w.x.array[:] = w_old.x.array[:]

    n_steps = int(round(T / dt_input))
    dt.value = T / n_steps

    for step in range(n_steps):
        current_time = (step + 1) * float(dt.value)
        time.value = current_time

        B_boundary.interpolate(B_numpy(current_time))
        u_boundary.interpolate(u_numpy(current_time))

        # Better initial guess for Newton
        w.sub(0).interpolate(B_numpy(current_time))
        w.sub(1).interpolate(u_numpy(current_time))
        w.sub(2).interpolate(lambda x: np.zeros_like(x[0]))

        problem.solve()

        w_old.x.array[:] = w.x.array[:]

    Bh, uh, ph = w.split()

    e_B = Bh - B_exact
    e_u = uh - u_exact
    e_p = ph 

    dx_error = ufl.dx(metadata={"quadrature_degree": 10})

    B_L2_form = fem.form(ufl.inner(e_B, e_B) * dx_error)
    B_curl_form = fem.form(ufl.inner(ufl.curl(e_B), ufl.curl(e_B)) * dx_error)

    u_L2_form = fem.form(ufl.inner(e_u, e_u) * dx_error)
    u_H1_form = fem.form(ufl.inner(ufl.grad(e_u), ufl.grad(e_u))* dx_error)
    u_L5_form = fem.form((ufl.inner(e_u, e_u))**2.5 * dx_error)

    grad_e_u = ufl.grad(e_u)
    grad_e_u_norm_squared = ufl.inner(grad_e_u, grad_e_u)

    u_W15_form = fem.form((grad_e_u_norm_squared**2.5) * dx_error)



    p_L2_form = fem.form(e_p * e_p * dx_error)

    B_L2_error = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(B_L2_form), op=MPI.SUM
    ))

    B_curl_error = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(B_curl_form), op=MPI.SUM
    ))

    u_L2_error = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(u_L2_form), op=MPI.SUM
    ))

    u_H1_error = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(u_H1_form), op=MPI.SUM
    ))
    u_L5_error = domain.comm.allreduce(
        fem.assemble_scalar(u_L5_form),
        op=MPI.SUM
    )**(1.0 / 5.0)

    u_W15_error = domain.comm.allreduce(
        fem.assemble_scalar(u_W15_form),
        op=MPI.SUM
    )**(1.0 / 5.0)

    p_L2_error = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(p_L2_form), op=MPI.SUM
    ))

    return h_value, float(dt.value), B_L2_error, B_curl_error, u_L2_error, u_H1_error, u_L5_error, u_W15_error, p_L2_error


T = 1.0
h0 = 0.866
dt0 = 0.25

cases = [
    (2, dt0, h0),
    (4, dt0 / 4.0, h0 / 2.0),
    (8, dt0 / 16.0, h0 / 4.0),
]

results = []

for n, dt_value, h_value in cases:
    print(f"Solving n={n}, h={h_value:.6f}, dt={dt_value:.6f}")
    results.append(solve_problem(n, dt_value, T, h_value))

print("")
print("Combined h-dt convergence table")
print(
    "h          dt         "
    "B L2 error        B L2 rate        "
    "curl B error      curl B rate      "
    "u L2 error        u L2 rate        "
    "u H1 error        u H1 rate        "
    "u L5 error        u L5 rate        "
    "u W15 error       u W15 rate       "
    "p L2 error        p L2 rate"
)

old_h = None
old_B_L2 = None
old_B_curl = None
old_u_L2 = None
old_u_H1 = None
old_u_L5 = None
old_u_W15 = None
old_p_L2 = None

for h, dt, B_L2, B_curl, u_L2, u_H1, u_L5, u_W15, p_L2 in results:
    if old_h is None:
        B_L2_rate = "-"
        B_curl_rate = "-"
        u_L2_rate = "-"
        u_H1_rate = "-"
        u_L5_rate = "-"
        u_W15_rate = "-"
        p_L2_rate = "-"
    else:
        B_L2_rate = np.log(old_B_L2 / B_L2) / np.log(old_h / h)
        B_curl_rate = np.log(old_B_curl / B_curl) / np.log(old_h / h)
        u_L2_rate = np.log(old_u_L2 / u_L2) / np.log(old_h / h)
        u_H1_rate = np.log(old_u_H1 / u_H1) / np.log(old_h / h)
        u_L5_rate = np.log(old_u_L5 / u_L5) / np.log(old_h / h)
        u_W15_rate = np.log(old_u_W15 / u_W15) / np.log(old_h / h)
        p_L2_rate = np.log(old_p_L2 / p_L2) / np.log(old_h / h)

    print(
        f"{h:<10.6f} {dt:<10.6f} "
        f"{B_L2:<17.6e} {B_L2_rate!s:<16} "
        f"{B_curl:<17.6e} {B_curl_rate!s:<16} "
        f"{u_L2:<17.6e} {u_L2_rate!s:<16} "
        f"{u_H1:<17.6e} {u_H1_rate!s:<16} "
        f"{u_L5:<17.6e} {u_L5_rate!s:<16} "
        f"{u_W15:<17.6e} {u_W15_rate!s:<16} "
        f"{p_L2:<17.6e} {p_L2_rate!s:<16}"
    )

    old_h = h
    old_B_L2 = B_L2
    old_B_curl = B_curl
    old_u_L2 = u_L2
    old_u_H1 = u_H1
    old_u_L5 = u_L5
    old_u_W15 = u_W15
    old_p_L2 = p_L2

  
