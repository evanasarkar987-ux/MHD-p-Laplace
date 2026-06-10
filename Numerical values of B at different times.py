from mpi4py import MPI
import numpy as np
import ufl
import matplotlib.pyplot as plt

from basix.ufl import element, mixed_element
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells


def evaluate_function_at_point(function, point, domain):
    tree = bb_tree(domain, domain.topology.dim)
    points = np.array([point], dtype=np.float64)

    candidate_cells = compute_collisions_points(tree, points)
    colliding_cells = compute_colliding_cells(domain, candidate_cells, points)
    cells = colliding_cells.links(0)

    if len(cells) == 0:
        return None

    value = function.eval(points, np.array([cells[0]], dtype=np.int32))
    return np.asarray(value).reshape(-1)


def solve_problem():
    n = 4
    dt_input = 0.025
    T = 1.0



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
    zero = fem.Constant(domain, 0.0)

    u_exact = ufl.as_vector((
        ufl.sin(np.pi * x[1]) * ufl.cos(time),
        ufl.sin(np.pi * x[2]) * ufl.sin(time),
        ufl.sin(np.pi * x[0]) * ufl.cos(2.0 * time),
    ))

    B_exact = ufl.as_vector((
        np.pi * ufl.sin(time) * ufl.sin(np.pi * x[0]) * ufl.cos(np.pi * x[1]),

        -np.pi * ufl.sin(time) * ufl.cos(np.pi * x[0]) * ufl.sin(np.pi * x[1])
        - np.pi * ufl.cos(time) * ufl.cos(np.pi * x[0]) * ufl.sin(np.pi * x[2]),

        np.pi * ufl.cos(time) * ufl.sin(np.pi * x[0]) * ufl.cos(np.pi * x[2]),
    ))

    u_t_exact = ufl.as_vector((
        -ufl.sin(np.pi * x[1]) * ufl.sin(time),
        ufl.sin(np.pi * x[2]) * ufl.cos(time),
        -2.0 * ufl.sin(np.pi * x[0]) * ufl.sin(2.0 * time),
    ))
    B_t_exact = ufl.as_vector((
        np.pi * ufl.cos(time) * ufl.sin(np.pi * x[0]) * ufl.cos(np.pi * x[1]),

        -np.pi * ufl.cos(time) * ufl.cos(np.pi * x[0]) * ufl.sin(np.pi * x[1])
        + np.pi * ufl.sin(time) * ufl.cos(np.pi * x[0]) * ufl.sin(np.pi * x[2]),

        -np.pi * ufl.sin(time) * ufl.sin(np.pi * x[0]) * ufl.cos(np.pi * x[2]),
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
            np.pi * np.sin(t) * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]),

            -np.pi * np.sin(t) * np.cos(np.pi * x[0]) * np.sin(np.pi * x[1])
            - np.pi * np.cos(t) * np.cos(np.pi * x[0]) * np.sin(np.pi * x[2]),

            np.pi * np.cos(t) * np.sin(np.pi * x[0]) * np.cos(np.pi * x[2]),
    ))



    def u_numpy(t):
        return lambda x: np.vstack((
            np.sin(np.pi * x[1]) * np.cos(t),
            np.sin(np.pi * x[2]) * np.sin(t),
            np.sin(np.pi * x[0]) * np.cos(2.0 * t),
        ))

    B_dofs = fem.locate_dofs_topological((W.sub(0), B_space), fdim, boundary_facets)
    u_dofs = fem.locate_dofs_topological((W.sub(1), u_space), fdim, boundary_facets)

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
        petsc_options_prefix="plot_B_direction_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-8,
            "snes_max_it": 50,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    w_old.sub(0).interpolate(B_numpy(0.0))
    w_old.sub(1).interpolate(u_numpy(0.0))
    w_old.sub(2).interpolate(lambda x: np.zeros_like(x[0]))
    w.x.array[:] = w_old.x.array[:]
    if T == 0.0:
        Bh = w.sub(0).collapse()
        return domain, Bh, T
    n_steps = int(round(T / dt_input))
    dt.value = T / n_steps

    for step in range(n_steps):
        current_time = (step + 1) * float(dt.value)
        time.value = current_time

        B_boundary.interpolate(B_numpy(current_time))
        u_boundary.interpolate(u_numpy(current_time))

        w.sub(0).interpolate(B_numpy(current_time))
        w.sub(1).interpolate(u_numpy(current_time))
        w.sub(2).interpolate(lambda x: np.zeros_like(x[0]))

        problem.solve()
        w_old.x.array[:] = w.x.array[:]

    Bh = w.sub(0).collapse()


    return domain, Bh, T


domain, Bh, T = solve_problem()

grid_n = 17
x_values = np.linspace(0.05, 0.95, grid_n)
y_values = np.linspace(0.05, 0.95, grid_n)

X, Y = np.meshgrid(x_values, y_values)

B1 = np.zeros_like(X)
B2 = np.zeros_like(X)

z_slice = 0.25

for i in range(grid_n):
    for j in range(grid_n):
        point = np.array([X[i, j], Y[i, j], z_slice], dtype=np.float64)
        B_value = evaluate_function_at_point(Bh, point, domain)


        if B_value is not None:
            B1[i, j] = B_value[0]
            B2[i, j] = B_value[1]


plt.figure(figsize=(5, 5))

plt.quiver(
    X,
    Y,
    B1,
    B2,
    color="red",
    angles="xy",
    scale_units="xy",
    scale=35,
    width=0.004,
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("t=1.0")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect("equal")
plt.grid(True)
plt.tight_layout()

plt.savefig("B_direction_plot_t_1_0.png", dpi=200)

print("Saved plot to B_direction_plot_t_1_0.png")
