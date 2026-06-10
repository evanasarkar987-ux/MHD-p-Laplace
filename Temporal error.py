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

    B_element = element("Nedelec 1st kind H(curl)", cell, 2)
    velocity_element = element("Lagrange", cell, 2, shape=(gdim,))
    pressure_element = element("Lagrange", cell, 1)

    W = fem.functionspace(
        domain,
        mixed_element([B_element, velocity_element, pressure_element]),
    )

    w = fem.Function(W)
    w_old = fem.Function(W)

    B, u, p = ufl.split(w)
    B_old, u_old, p_old = ufl.split(w_old)

    C, v, q = ufl.TestFunctions(W)

    nu = fem.Constant(domain, 1.0)
    dt = fem.Constant(domain, dt_input)
    time = fem.Constant(domain, 0.0)

    x = ufl.SpatialCoordinate(domain)
    p_value = 3
    eps = fem.Constant(domain, 1.0e-8)
    B_shape = ufl.as_vector((
        ufl.sin(np.pi * x[1]),
        ufl.sin(np.pi * x[2]),
        ufl.sin(np.pi * x[0]),
    ))

    u_shape = ufl.as_vector((
        ufl.sin(np.pi * x[0]) * ufl.cos(np.pi * x[1]) * ufl.cos(np.pi * x[2]),
        -ufl.cos(np.pi * x[0]) * ufl.sin(np.pi * x[1]) * ufl.cos(np.pi * x[2]),
        0.0 * x[2],
    ))

    p_shape = (
        ufl.sin(np.pi * x[0])
        * ufl.sin(np.pi * x[1])
        * ufl.sin(np.pi * x[2])
    )

    B_exact = ufl.exp(-time) * B_shape
    u_exact = ufl.exp(-time) * u_shape
    p_exact = ufl.exp(-time) * p_shape

    B_t_exact = -B_exact
    u_t_exact = -u_exact
    grad_exact = ufl.grad(u_exact)
    grad_exact_norm = ufl.sqrt(ufl.inner(grad_exact, grad_exact)+eps)
    grad_u = ufl.grad(u)
    grad_u_norm = ufl.sqrt(ufl.inner(grad_u, grad_u)+eps)
    F_current = (
        ufl.inner((B - B_old) / dt, C) * ufl.dx
        + ufl.inner(ufl.curl(B), ufl.curl(C)) * ufl.dx
        - ufl.inner(ufl.cross(u, B), ufl.curl(C)) * ufl.dx
        + ufl.inner((u - u_old) / dt, v) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        + nu * grad_u_norm**(p_value-2.0) * ufl.inner(grad_u, ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.cross(B, ufl.curl(B)), v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
    )

    F_exact = (
        ufl.inner(B_t_exact, C) * ufl.dx
        + ufl.inner(ufl.curl(B_exact), ufl.curl(C)) * ufl.dx
        - ufl.inner(ufl.cross(u_exact, B_exact), ufl.curl(C)) * ufl.dx
        + ufl.inner(u_t_exact, v) * ufl.dx
        + ufl.inner(ufl.grad(u_exact) * u_exact, v) * ufl.dx
        + nu * grad_exact_norm**(p_value-2.0) * ufl.inner(grad_exact, ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.cross(B_exact, ufl.curl(B_exact)), v) * ufl.dx
        - p_exact * ufl.div(v) * ufl.dx
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
    Q0, _ = W.sub(2).collapse()

    B_boundary = fem.Function(B_space)
    u_boundary = fem.Function(u_space)
    p_zero = fem.Function(Q0)
    p_zero.x.array[:] = 0.0

    def B_exact_numpy(t):
        return lambda x: np.exp(-t) * np.vstack((
            np.sin(np.pi * x[1]),
            np.sin(np.pi * x[2]),
            np.sin(np.pi * x[0]),
        ))

    def u_exact_numpy(t):
        return lambda x: np.exp(-t) * np.vstack((
            np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]) * np.cos(np.pi * x[2]),
            -np.cos(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.cos(np.pi * x[2]),
            np.zeros_like(x[2]),
        ))

    def p_exact_numpy(t):
        return lambda x: np.exp(-t) * (
            np.sin(np.pi * x[0])
            * np.sin(np.pi * x[1])
            * np.sin(np.pi * x[2])
        )

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
        (W.sub(2), Q0),
        lambda x: np.logical_and.reduce((
            np.isclose(x[0], 0.0),
            np.isclose(x[1], 0.0),
            np.isclose(x[2], 0.0),
        )),
    )

    bc_B = fem.dirichletbc(B_boundary, B_dofs, W.sub(0))
    bc_u = fem.dirichletbc(u_boundary, u_dofs, W.sub(1))
    bc_pressure = fem.dirichletbc(p_zero, pressure_dofs, W.sub(2))

    problem = NonlinearProblem(
        residual,
        w,
        bcs=[bc_B, bc_u, bc_pressure],
        petsc_options_prefix=f"mhd_time_{n}_{dt_input}_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-9,
            "snes_atol": 1e-9,
            "snes_max_it": 30,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_error_if_not_converged": True,
            "ksp_error_if_not_converged": True,
        },
    )

    # Initial condition at t = 0
    time.value = 0.0
    w_old.sub(0).interpolate(B_exact_numpy(0.0))
    w_old.sub(1).interpolate(u_exact_numpy(0.0))
    w_old.sub(2).interpolate(p_exact_numpy(0.0))
    w.x.array[:] = w_old.x.array[:]

    n_steps = int(round(T / dt_input))
    dt.value = T / n_steps

    for step in range(n_steps):
        current_time = (step + 1) * float(dt.value)
        time.value = current_time

        B_boundary.interpolate(B_exact_numpy(current_time))
        u_boundary.interpolate(u_exact_numpy(current_time))
        w.sub(0).interpolate(B_exact_numpy(current_time))
        w.sub(1).interpolate(u_exact_numpy(current_time))
        w.sub(2).interpolate(p_exact_numpy(current_time))
        problem.solve()

        w_old.x.array[:] = w.x.array[:]

    Bh, uh, ph = w.split()

    e_B = Bh - B_exact
    e_u = uh - u_exact
    e_p = ph - p_exact

    dx_error = ufl.dx(metadata={"quadrature_degree": 10})

    B_L2_form = fem.form(ufl.inner(e_B, e_B) * dx_error)
    B_curl_form = fem.form(ufl.inner(ufl.curl(e_B), ufl.curl(e_B)) * dx_error)
    u_L2_form = fem.form(ufl.inner(e_u, e_u) * dx_error)
    u_H1_form = fem.form(
         ufl.inner(ufl.grad(e_u), ufl.grad(e_u))
        * dx_error
    )
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
    p_L2_error = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(p_L2_form), op=MPI.SUM
    ))

    return float(dt.value), B_L2_error, B_curl_error, u_L2_error, u_H1_error, p_L2_error

n = 16
T = 1.6
dt_values = [0.8, 0.4, 0.2]


results = []

for dt_value in dt_values:
    print(f"Solving with dt = {dt_value}")
    results.append(solve_problem(n, dt_value, T))

print("")
print("Time convergence table")
print(
    "dt         "
    "B L2 error        B L2 rate        "
    "curl B error      curl B rate      "
    "u L2 error        u L2 rate        "
    "u H1 error        u H1 rate        "
    "p L2 error        p L2 rate"
)

old_dt = None
old_B_L2 = None
old_B_curl = None
old_u_L2 = None
old_u_H1 = None
old_p_L2 = None

for dt, B_L2, B_curl, u_L2, u_H1, p_L2 in results:
    if old_dt is None:
        B_L2_rate = "-"
        B_curl_rate = "-"
        u_L2_rate = "-"
        u_H1_rate = "-"
        p_L2_rate = "-"
    else:
        B_L2_rate = np.log(old_B_L2 / B_L2) / np.log(old_dt / dt)
        B_curl_rate = np.log(old_B_curl / B_curl) / np.log(old_dt / dt)
        u_L2_rate = np.log(old_u_L2 / u_L2) / np.log(old_dt / dt)
        u_H1_rate = np.log(old_u_H1 / u_H1) / np.log(old_dt / dt)
        p_L2_rate = np.log(old_p_L2 / p_L2) / np.log(old_dt / dt)

    print(
        f"{dt:<10.5f} "
        f"{B_L2:<17.6e} {B_L2_rate!s:<16} "
        f"{B_curl:<17.6e} {B_curl_rate!s:<16} "
        f"{u_L2:<17.6e} {u_L2_rate!s:<16} "
        f"{u_H1:<17.6e} {u_H1_rate!s:<16} "
        f"{p_L2:<17.6e} {p_L2_rate!s:<16}"
    )

    old_dt = dt
    old_B_L2 = B_L2
    old_B_curl = B_curl
    old_u_L2 = u_L2
    old_u_H1 = u_H1
    old_p_L2 = p_L2
