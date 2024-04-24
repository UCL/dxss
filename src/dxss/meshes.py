from importlib.metadata import version

import gmsh
import numpy as np
from dolfinx.cpp.io import perm_gmsh
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import (
    CellType,
    GhostMode,
    compute_incident_entities,
    create_mesh,
    locate_entities,
    refine,
)
from mpi4py import MPI
from packaging.version import Version

GM = GhostMode.shared_facet
eta = 0.6


def create_initial_mesh_convex(init_h_scale=1.0):  # noqa: PLR0915
    # TODO: Need to understand why there is a an import and subfunction here.
    # Looks like create_mesh could be global level. The following functions have
    # a **lot** of code overlap with this function.
    #
    # - PLR0915 Too many statements
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", init_h_scale)
    proc = MPI.COMM_WORLD.rank
    bnd_marker = 1
    omega_marker = 1
    b_withoutomega_marker = 2
    rest_marker = 3
    if proc == 0:
        # We create one rectangle for each subdomain

        r1 = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1, tag=1)
        r2 = gmsh.model.occ.addRectangle(0.1, 0.25, 0, 0.8, 0.75, tag=2)
        r3 = gmsh.model.occ.cut([(2, r1)], [(2, r2)], tag=3)

        print("r3 = ", r3)
        gmsh.model.occ.addRectangle(0.1, 0.25, 0, 0.8, 0.7, tag=4)

        # We fuse the two rectangles and keep the interface between them
        gmsh.model.occ.fragment([(2, 3)], [(2, 4)])

        tmp = gmsh.model.occ.addRectangle(0.1, 0.95, 0, 0.8, 0.05, tag=6)
        gmsh.model.occ.fragment([(2, 3)], [(2, 4), (2, tmp)])

        gmsh.model.occ.synchronize()

        # for surface in gmsh.model.getEntities(dim=2):

        # Mark the top (2) and bottom (1) rectangle
        print(len(gmsh.model.getEntities(dim=2)))
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            if np.allclose(com, [0.5, 0.6, 0]):
                gmsh.model.addPhysicalGroup(2, [surface[1]], b_withoutomega_marker)
            elif np.allclose(com, [0.5, 0.975, 0]):
                gmsh.model.addPhysicalGroup(2, [surface[1]], rest_marker)
            else:
                gmsh.model.addPhysicalGroup(2, [surface[1]], omega_marker)
        #    if np.allclose(com, [0.5,0.25, 0]):
        # Tag the left boundary
        bnd_square = []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if (
                np.isclose(com[0], 0)
                or np.isclose(com[0], 1)
                or np.isclose(com[1], 0)
                or np.isclose(com[1], 1)
            ):
                bnd_square.append(line[1])
        gmsh.model.addPhysicalGroup(1, bnd_square, bnd_marker)
        gmsh.model.mesh.generate(2)
        gmsh.write("mesh.msh")
        gmsh.finalize()

    import meshio

    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:, :2] if prune_z else mesh.points
        return meshio.Mesh(
            points=points,
            cells={cell_type: cells},
            cell_data={"name_to_read": [cell_data]},
        )

    if proc == 0:
        # Read in mesh
        msh = meshio.read("mesh.msh")

        # Create and save one file for the mesh, and one file for the facets
        triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
        line_mesh = create_mesh(msh, "line", prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf", line_mesh)

    # for i in range(n_ref):

    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid", ghost_mode=GM)
        xdmf.read_meshtags(mesh, name="Grid")
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
        xdmf.read_meshtags(mesh, name="Grid")
    return mesh


def get_mesh_convex(n_ref, init_h_scale=1.0):
    meshes = []
    for j in range(n_ref):
        h_scale_j = init_h_scale / (2**j)
        meshes.append(create_initial_mesh_convex(init_h_scale=h_scale_j))
    return meshes


def get_mesh_hierarchy(n_ref, init_h_scale=1.0):  # noqa: PLR0915
    # TODO: This is very WET from create_inital_mesh_complex above. We need to
    # find some way to not repeat this function. Also fix PLR0915 and remove the
    # suppression.
    #
    # - PLR0915 Too many statements
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", init_h_scale)
    proc = MPI.COMM_WORLD.rank
    bnd_marker = 1
    omega_marker = 1
    b_withoutomega_marker = 2
    rest_marker = 3
    if proc == 0:
        # We create one rectangle for each subdomain

        r1 = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1, tag=1)
        r2 = gmsh.model.occ.addRectangle(0.1, 0.25, 0, 0.8, 0.75, tag=2)
        r3 = gmsh.model.occ.cut([(2, r1)], [(2, r2)], tag=3)

        print("r3 = ", r3)
        gmsh.model.occ.addRectangle(0.1, 0.25, 0, 0.8, 0.7, tag=4)

        # We fuse the two rectangles and keep the interface between them
        gmsh.model.occ.fragment([(2, 3)], [(2, 4)])

        tmp = gmsh.model.occ.addRectangle(0.1, 0.95, 0, 0.8, 0.05, tag=6)
        gmsh.model.occ.fragment([(2, 3)], [(2, 4), (2, tmp)])

        gmsh.model.occ.synchronize()

        # for surface in gmsh.model.getEntities(dim=2):

        # Mark the top (2) and bottom (1) rectangle
        print(len(gmsh.model.getEntities(dim=2)))
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            if np.allclose(com, [0.5, 0.6, 0]):
                gmsh.model.addPhysicalGroup(2, [surface[1]], b_withoutomega_marker)
            elif np.allclose(com, [0.5, 0.975, 0]):
                gmsh.model.addPhysicalGroup(2, [surface[1]], rest_marker)
            else:
                gmsh.model.addPhysicalGroup(2, [surface[1]], omega_marker)
        #    if np.allclose(com, [0.5,0.25, 0]):
        # Tag the left boundary
        bnd_square = []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if (
                np.isclose(com[0], 0)
                or np.isclose(com[0], 1)
                or np.isclose(com[1], 0)
                or np.isclose(com[1], 1)
            ):
                bnd_square.append(line[1])
        gmsh.model.addPhysicalGroup(1, bnd_square, bnd_marker)
        gmsh.model.mesh.generate(2)
        gmsh.write("mesh.msh")
    gmsh.finalize()

    import meshio

    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:, :2] if prune_z else mesh.points
        return meshio.Mesh(
            points=points,
            cells={cell_type: cells},
            cell_data={"name_to_read": [cell_data]},
        )

    if proc == 0:
        # Read in mesh
        msh = meshio.read("mesh.msh")

        # Create and save one file for the mesh, and one file for the facets
        triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
        line_mesh = create_mesh(msh, "line", prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf", line_mesh)

    # for i in range(n_ref):

    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        print("ghost_mode = GhostMode.shared_facet")
        mesh = xdmf.read_mesh(name="Grid", ghost_mode=GM)
        xdmf.read_meshtags(mesh, name="Grid")
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
        xdmf.read_meshtags(mesh, name="Grid")

    mesh_hierarchy = []
    mesh_hierarchy.append(mesh)

    def refine_all(x):
        return x[0] >= 0

    for _i in range(n_ref):
        mesh.topology.create_entities(1)
        cells = locate_entities(mesh, mesh.topology.dim, refine_all)
        edges = compute_incident_entities(mesh, cells, 2, 1)
        mesh = refine(mesh, edges, redistribute=True)
        mesh_hierarchy.append(mesh)
    return mesh_hierarchy


def get_mesh_hierarchy_nonconvex(n_ref, init_h_scale=1.0):  # noqa: PLR0915
    # TODO: remove suppression of PLR0915 when refactored
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", init_h_scale)
    proc = MPI.COMM_WORLD.rank
    bnd_marker = 1
    omega_marker = 1
    b_withoutomega_marker = 2
    rest_marker = 3
    if proc == 0:
        # We create one rectangle for each subdomain

        gmsh.model.occ.addRectangle(0.25, 0.05, 0, 0.5, 0.45, tag=1)
        gmsh.model.occ.addRectangle(0.125, 0.05, 0, 0.75, 0.9, tag=2)

        gmsh.model.occ.addRectangle(0, 0, 0, 1, 1, tag=3)
        gmsh.model.occ.fragment([(2, 3)], [(2, 1), (2, 2)])

        gmsh.model.occ.synchronize()
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            if np.allclose(com, [0.5, 0.275, 0]):
                gmsh.model.addPhysicalGroup(2, [surface[1]], omega_marker)
            elif np.allclose(com, [0.5, 0.5, 0]):
                gmsh.model.addPhysicalGroup(2, [surface[1]], b_withoutomega_marker)
            else:
                gmsh.model.addPhysicalGroup(2, [surface[1]], rest_marker)

        #    if np.allclose(com, [0.5,0.25, 0]):
        # Tag the left boundary
        bnd_square = []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if (
                np.isclose(com[0], 0)
                or np.isclose(com[0], 1)
                or np.isclose(com[1], 0)
                or np.isclose(com[1], 1)
            ):
                bnd_square.append(line[1])
        gmsh.model.addPhysicalGroup(1, bnd_square, bnd_marker)
        gmsh.model.mesh.generate(2)
        gmsh.write("mesh.msh")
    gmsh.finalize()

    import meshio

    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:, :2] if prune_z else mesh.points
        return meshio.Mesh(
            points=points,
            cells={cell_type: cells},
            cell_data={"name_to_read": [cell_data]},
        )

    if proc == 0:
        # Read in mesh
        msh = meshio.read("mesh.msh")
        # Create and save one file for the mesh, and one file for the facets
        triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
        line_mesh = create_mesh(msh, "line", prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf", line_mesh)

    # for i in range(n_ref):
    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid", ghost_mode=GM)
        xdmf.read_meshtags(mesh, name="Grid")
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
        xdmf.read_meshtags(mesh, name="Grid")

    mesh_hierarchy = []
    mesh_hierarchy.append(mesh)

    def refine_all(x):
        return x[0] >= 0

    for _i in range(n_ref):
        mesh.topology.create_entities(1)
        cells = locate_entities(mesh, mesh.topology.dim, refine_all)
        edges = compute_incident_entities(mesh, cells, 2, 1)
        print(edges)
        mesh = refine(mesh, edges, redistribute=True)
        mesh_hierarchy.append(mesh)

    return mesh_hierarchy


def get_mesh_hierarchy_fitted_disc(n_ref, eta, h_init=1.25):  # noqa: PLR0915
    # TODO: remove suppression of PLR0915 when refactored
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", h_init)
    proc = MPI.COMM_WORLD.rank
    bnd_marker = 1

    y_eta = eta - 0.25
    0.95 - y_eta
    if proc == 0:
        # We create one rectangle for each subdomain

        r1 = gmsh.model.occ.addRectangle(0, 0, 0, 1, eta, tag=1)
        r2 = gmsh.model.occ.addRectangle(0.1, 0.25, 0, 0.8, y_eta, tag=2)
        gmsh.model.occ.cut([(2, r1)], [(2, r2)], tag=3)
        b_minus = gmsh.model.occ.addRectangle(0.1, 0.25, 0, 0.8, y_eta, tag=4)
        b_plus = gmsh.model.occ.addRectangle(0.1, y_eta, 0, 0.8, (0.95 - y_eta), tag=5)
        top_remainder = gmsh.model.occ.addRectangle(
            0.0,
            y_eta,
            0,
            1.0,
            (1.0 - y_eta),
            tag=6,
        )

        gmsh.model.occ.fragment(
            [(2, 3)],
            [(2, b_minus), (2, b_plus), (2, top_remainder)],
        )

        # We fuse the two rectangles and keep the interface between them

        gmsh.model.occ.synchronize()

        # for surface in gmsh.model.getEntities(dim=2):

        # Mark the top (2) and bottom (1) rectangle
        print(len(gmsh.model.getEntities(dim=2)))

        its = 1
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            gmsh.model.addPhysicalGroup(2, [surface[1]], its)
            its += 1

        # Tag the left boundary
        bnd_square = []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if (
                np.isclose(com[0], 0)
                or np.isclose(com[0], 1)
                or np.isclose(com[1], 0)
                or np.isclose(com[1], 1)
            ):
                bnd_square.append(line[1])
        gmsh.model.addPhysicalGroup(1, bnd_square, bnd_marker)
        gmsh.model.mesh.generate(2)
        gmsh.write("mesh.msh")
        gmsh.finalize()

    import meshio

    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:, :2] if prune_z else mesh.points
        return meshio.Mesh(
            points=points,
            cells={cell_type: cells},
            cell_data={"name_to_read": [cell_data]},
        )

    if proc == 0:
        # Read in mesh
        msh = meshio.read("mesh.msh")

        # Create and save one file for the mesh, and one file for the facets
        triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
        line_mesh = create_mesh(msh, "line", prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf", line_mesh)

    # for i in range(n_ref):

    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid", ghost_mode=GM)
        xdmf.read_meshtags(mesh, name="Grid")
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
        xdmf.read_meshtags(mesh, name="Grid")

    mesh_hierarchy = []
    mesh_hierarchy.append(mesh)

    def refine_all(x):
        return x[0] >= 0

    for _i in range(n_ref):
        mesh.topology.create_entities(1)
        cells = locate_entities(mesh, mesh.topology.dim, refine_all)
        edges = compute_incident_entities(mesh, cells, 2, 1)
        print(edges)
        mesh = refine(mesh, edges, redistribute=True)
        mesh_hierarchy.append(mesh)
    return mesh_hierarchy


def get_mesh_inclusion(h_init=1.25, order=2):  # noqa: PLR0915
    # TODO: remove suppression of PLR0915 when refactored
    cell_type = CellType.triangle
    gmsh.initialize()
    gmsh.model.add("inclm")
    gmsh.option.setNumber("Mesh.MeshSizeFactor", h_init)
    bnd_marker = 1

    # if proc == 0:
    # We create one rectangle for each subdomain

    if MPI.COMM_WORLD.rank == 0:
        r1 = gmsh.model.occ.addRectangle(-1.5, -1.5, 0, 3, 3, tag=1)
        r2 = gmsh.model.occ.addRectangle(-1.25, -1.25, 0, 2.5, 2.75, tag=2)
        gmsh.model.occ.cut([(2, r1)], [(2, r2)], tag=3)

        target_dom = gmsh.model.occ.addRectangle(-0.5, -0.5, 0, 1.0, 1.0)
        b_disk = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, 1.0, 1.0, tag=2)
        gmsh.model.occ.synchronize()
        remainder = gmsh.model.occ.addRectangle(-1.25, -1.25, 0, 2.5, 2.75)
        gmsh.model.occ.fragment(
            [(2, 3)],
            [(2, target_dom), (2, b_disk), (2, remainder)],
        )

        gmsh.model.occ.synchronize()

        print(len(gmsh.model.getEntities(dim=2)))

        its = 1
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            gmsh.model.addPhysicalGroup(2, [surface[1]], its)
            its += 1

        # Tag the left boundary
        bnd_square = []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if (
                np.isclose(com[0], 0)
                or np.isclose(com[0], 1)
                or np.isclose(com[1], 0)
                or np.isclose(com[1], 1)
            ):
                bnd_square.append(line[1])
        gmsh.model.addPhysicalGroup(1, bnd_square, bnd_marker)
        gmsh.model.mesh.generate(2)
        gmsh.write("mesh.msh")

    if cell_type == CellType.quadrilateral:
        gmsh.model.mesh.recombine()
    gmsh.model.mesh.setOrder(order)
    idx, points, _ = gmsh.model.mesh.getNodes()
    ls_points_2d = [point for i, point in enumerate(points) if (i + 1) % 3 != 0]
    ls_points_2d = np.array(ls_points_2d)
    points = ls_points_2d.reshape(-1, 2)
    idx -= 1
    srt = np.argsort(idx)
    if np.all(idx[srt] == np.arange(len(idx))):
        msg = "idx not sequential"
        raise RuntimeError(msg)  # TODO: check
    x = points[srt]

    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
    (
        name,
        dim,
        order,
        num_nodes,
        local_coords,
        num_first_order_nodes,
    ) = gmsh.model.mesh.getElementProperties(element_types[0])

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    if cell_type == CellType.triangle:
        gmsh_cell_id = gmsh.model.mesh.getElementType("triangle", order)
    elif cell_type == CellType.quadrilateral:
        gmsh_cell_id = gmsh.model.mesh.getElementType("quadrangle", order)
    gmsh.finalize()

    cells = cells[:, perm_gmsh(cell_type, cells.shape[1])]
    msh = create_mesh(
        MPI.COMM_WORLD,
        cells,
        x,
        gmshio.ufl_mesh(gmsh_cell_id, x.shape[1]),
    )

    with XDMFFile(msh.comm, "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
    return msh


def get_mesh_inclusion_square(  # noqa: PLR0915
    h_init=1.25,
    x_l=-1.25,  # TODO: points could be passed as a tuple
    x_r=1.25,
    y_l=-1.25,
    y_r=1.25,
    eta=1.25,
):
    # TODO: this function has too many statements. Try to reduce the complexity
    # on refactoring and remove the suppression of PLR0915.
    cell_type = CellType.triangle
    gmsh.initialize()
    gmsh.model.add("inclm")
    gmsh.option.setNumber("Mesh.MeshSizeFactor", h_init)
    bnd_marker = 1

    if MPI.COMM_WORLD.rank == 0:
        r1 = gmsh.model.occ.addRectangle(-1.5, -1.5, 0, 3, 1.5 + eta, tag=1)
        r2 = gmsh.model.occ.addRectangle(-1.25, -1.25, 0, 2.5, 1.15 + eta, tag=2)
        gmsh.model.occ.cut([(2, r1)], [(2, r2)], tag=3)

        if eta < 1.5:
            target_dom_minus = gmsh.model.occ.addRectangle(
                x_l,
                y_l,
                0,
                x_r - x_l,
                eta - y_l,
            )
            target_dom_plus = gmsh.model.occ.addRectangle(
                x_l,
                eta,
                0,
                x_r - x_l,
                y_r - eta,
            )
            mu_inner = gmsh.model.occ.addRectangle(-0.75, -0.75, 0, 1.5, 1.5)
            remainder_top = gmsh.model.occ.addRectangle(-1.5, eta, 0, 3.0, 1.5 - eta)
            gmsh.model.occ.synchronize()
            gmsh.model.occ.fragment(
                [(2, 3)],
                [
                    (2, target_dom_minus),
                    (2, mu_inner),
                    (2, target_dom_plus),
                    (2, remainder_top),
                ],
            )

            gmsh.model.occ.synchronize()

        else:
            target_dom = gmsh.model.occ.addRectangle(x_l, y_l, 0, x_r - x_l, y_r - y_l)
            mu_inner = gmsh.model.occ.addRectangle(-0.75, -0.75, 0, 1.5, 1.5)
            remainder = gmsh.model.occ.addRectangle(-1.25, -1.25, 0, 2.5, 2.75)
            gmsh.model.occ.synchronize()
            gmsh.model.occ.fragment(
                [(2, 3)],
                [(2, target_dom), (2, mu_inner), (2, remainder)],
            )

        gmsh.model.occ.synchronize()

        print(len(gmsh.model.getEntities(dim=2)))

        its = 1
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            gmsh.model.addPhysicalGroup(2, [surface[1]], its)
            its += 1

        # Tag the left boundary
        bnd_square = []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if (
                np.isclose(com[0], 0)
                or np.isclose(com[0], 1)
                or np.isclose(com[1], 0)
                or np.isclose(com[1], 1)
            ):
                bnd_square.append(line[1])
        gmsh.model.addPhysicalGroup(1, bnd_square, bnd_marker)
        gmsh.model.mesh.generate(2)
        gmsh.write("mesh.msh")

    if cell_type == CellType.quadrilateral:
        gmsh.model.mesh.recombine()
    order = 1
    gmsh.model.mesh.setOrder(order)
    idx, points, _ = gmsh.model.mesh.getNodes()
    ls_points_2d = [point for i, point in enumerate(points) if (i + 1) % 3 != 0]
    ls_points_2d = np.array(ls_points_2d)
    points = ls_points_2d.reshape(-1, 2)
    idx -= 1
    srt = np.argsort(idx)
    if np.all(idx[srt] == np.arange(len(idx))):
        msg = "idx not sequential"
        raise RuntimeError(msg)  # TODO: check
    x = points[srt]

    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
    (
        name,
        dim,
        order,
        num_nodes,
        local_coords,
        num_first_order_nodes,
    ) = gmsh.model.mesh.getElementProperties(element_types[0])

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    if cell_type == CellType.triangle:
        gmsh_cell_id = gmsh.model.mesh.getElementType("triangle", order)
    elif cell_type == CellType.quadrilateral:
        gmsh_cell_id = gmsh.model.mesh.getElementType("quadrangle", order)
    gmsh.finalize()

    cells = cells[:, perm_gmsh(cell_type, cells.shape[1])]
    msh = create_mesh(
        MPI.COMM_WORLD,
        cells,
        x,
        ufl_mesh_from_gmsh(  # noqa: F821
            gmshio.ufl_mesh(gmsh_cell_id, x.shape[1]),
        ),  # TODO: this function is undefined!! (old dolfinx api?)
    )

    with XDMFFile(msh.comm, "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
    return msh


def get_mesh_bottom_data(h_init=1.25, eta=0.6):
    # TODO: this function has too many statements. Try to reduce the complexity
    # on refactoring and remove the suppression of PLR0915.
    cell_type = CellType.triangle
    gmsh.initialize()
    gmsh.model.add("inclm")
    gmsh.option.setNumber("Mesh.MeshSizeFactor", h_init)

    if MPI.COMM_WORLD.rank == 0:
        omega = gmsh.model.occ.addRectangle(0.0, 0.0, 0, 1.0, 0.25, tag=1)

        middle_left = gmsh.model.occ.addRectangle(0, 0.25, 0, 0.25, eta - 0.25, tag=2)
        middle_right = gmsh.model.occ.addRectangle(
            0.75,
            0.25,
            0,
            0.25,
            eta - 0.25,
            tag=3,
        )
        b_minus = gmsh.model.occ.addRectangle(0.25, 0.25, 0, 0.5, eta - 0.25, tag=4)

        b_plus = gmsh.model.occ.addRectangle(0.25, eta, 0, 0.5, 0.9 - eta, tag=5)
        top_left = gmsh.model.occ.addRectangle(0, eta, 0, 0.25, 1.0 - eta, tag=6)
        top_middle = gmsh.model.occ.addRectangle(0.25, eta, 0, 0.5, 1.0 - eta, tag=7)
        top_right = gmsh.model.occ.addRectangle(0.75, eta, 0, 0.25, 1.0 - eta, tag=8)

        gmsh.model.occ.synchronize()
        gmsh.model.occ.fragment(
            [(2, omega)],
            [
                (2, b_minus),
                (2, middle_left),
                (2, middle_right),
                (2, b_plus),
                (2, top_left),
                (2, top_middle),
                (2, top_right),
            ],
        )

        gmsh.model.occ.synchronize()

        print(len(gmsh.model.getEntities(dim=2)))

        its = 1
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            gmsh.model.addPhysicalGroup(2, [surface[1]], its)
            its += 1

        # Tag the left boundary
        # for line in gmsh.model.getEntities(dim=1):
        #    if np.isclose(com[0], 0) or np.isclose(com[0], 1) or np.isclose(com[1], 0) or  np.isclose(com[1], 1):
        gmsh.model.mesh.generate(2)
        gmsh.write("mesh.msh")

    if cell_type == CellType.quadrilateral:
        gmsh.model.mesh.recombine()
    order = 1
    gmsh.model.mesh.setOrder(order)
    idx, points, _ = gmsh.model.mesh.getNodes()
    ls_points_2d = [point for i, point in enumerate(points) if (i + 1) % 3 != 0]
    ls_points_2d = np.array(ls_points_2d)
    points = ls_points_2d.reshape(-1, 2)
    idx -= 1
    srt = np.argsort(idx)
    if np.all(idx[srt] == np.arange(len(idx))):
        msg = "idx not sequential"
        raise RuntimeError(msg)  # TODO: check
    x = points[srt]

    element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)
    (
        name,
        dim,
        order,
        num_nodes,
        local_coords,
        num_first_order_nodes,
    ) = gmsh.model.mesh.getElementProperties(element_types[0])

    cells = node_tags[0].reshape(-1, num_nodes) - 1
    if cell_type == CellType.triangle:
        gmsh_cell_id = gmsh.model.mesh.getElementType("triangle", order)
    elif cell_type == CellType.quadrilateral:
        gmsh_cell_id = gmsh.model.mesh.getElementType("quadrangle", order)
    gmsh.finalize()

    cells = cells[:, perm_gmsh(cell_type, cells.shape[1])]
    msh = create_mesh(
        MPI.COMM_WORLD,
        cells,
        x,
        gmshio.ufl_mesh(gmsh_cell_id, x.shape[1]),
    )

    with XDMFFile(msh.comm, "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
    return msh


def get_mesh_data_all_around(n_ref, init_h_scale=1.0):  # noqa: PLR0915
    # TODO: this function has too many statements. Try to reduce the complexity
    # on refactoring and remove the suppression of PLR0915.
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", init_h_scale)
    proc = MPI.COMM_WORLD.rank
    bnd_marker = 1
    if proc == 0:
        # We create one rectangle for each subdomain

        r1 = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1, tag=1)
        r2 = gmsh.model.occ.addRectangle(0.2, 0.2, 0, 0.6, 0.6, tag=2)
        r3 = gmsh.model.occ.cut([(2, r1)], [(2, r2)], tag=3)

        print("r3 = ", r3)
        gmsh.model.occ.addRectangle(0.2, 0.2, 0, 0.6, 0.6, tag=4)

        # We fuse the two rectangles and keep the interface between them
        gmsh.model.occ.fragment([(2, 3)], [(2, 4)])

        gmsh.model.occ.synchronize()

        # for surface in gmsh.model.getEntities(dim=2):

        # Mark the top (2) and bottom (1) rectangle

        its = 1
        for surface in gmsh.model.getEntities(dim=2):
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            print(com)
            gmsh.model.addPhysicalGroup(2, [surface[1]], its)
            its += 1

        # Tag the left boundary
        bnd_square = []
        for line in gmsh.model.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(line[0], line[1])
            if (
                np.isclose(com[0], 0)
                or np.isclose(com[0], 1)
                or np.isclose(com[1], 0)
                or np.isclose(com[1], 1)
            ):
                bnd_square.append(line[1])
        gmsh.model.addPhysicalGroup(1, bnd_square, bnd_marker)
        gmsh.model.mesh.generate(2)
        gmsh.write("mesh.msh")
    gmsh.finalize()

    import meshio

    def create_mesh(mesh, cell_type, prune_z=False):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
        points = mesh.points[:, :2] if prune_z else mesh.points
        return meshio.Mesh(
            points=points,
            cells={cell_type: cells},
            cell_data={"name_to_read": [cell_data]},
        )

    if proc == 0:
        # Read in mesh
        msh = meshio.read("mesh.msh")

        # Create and save one file for the mesh, and one file for the facets
        triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
        line_mesh = create_mesh(msh, "line", prune_z=True)
        meshio.write("mesh.xdmf", triangle_mesh)
        meshio.write("mt.xdmf", line_mesh)

    # for i in range(n_ref):

    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
        print("ghost_mode = GhostMode.shared_facet")
        mesh = xdmf.read_mesh(name="Grid", ghost_mode=GM)
        xdmf.read_meshtags(mesh, name="Grid")
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "r") as xdmf:
        xdmf.read_meshtags(mesh, name="Grid")

    mesh_hierarchy = []
    mesh_hierarchy.append(mesh)

    def refine_all(x):
        return x[0] >= 0

    for _i in range(n_ref):
        mesh.topology.create_entities(1)
        cells = locate_entities(mesh, mesh.topology.dim, refine_all)
        # Call signature for compute_incident_entities changed to accept mesh topology
        # rather than mesh in dolfinx v0.7 so set relevant argument accordingly based on
        # installed version
        if Version(version("fenics-dolfinx")) < Version("0.7"):
            mesh_or_topology = mesh
        else:
            mesh_or_topology = mesh.topology
        edges = compute_incident_entities(mesh_or_topology, cells, 2, 1)
        mesh = refine(mesh, edges, redistribute=True)
        mesh_hierarchy.append(mesh)
    return mesh_hierarchy


def get_3d_mesh_data_all_around(n_ref, init_h_scale=1.0):
    # if proc == 0:
    gdim = 3
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeFactor", init_h_scale)
    # We create one cube for each subdomain

    r1 = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, tag=1)

    r2 = gmsh.model.occ.addBox(0.2, 0.2, 0.2, 0.6, 0.6, 0.6, tag=2)
    gmsh.model.occ.cut([(3, r1)], [(3, r2)], tag=3)
    gmsh.model.occ.addBox(0.2, 0.2, 0.2, 0.6, 0.6, 0.6, tag=4)

    # We fuse the two boxes and keep the interface between them
    ov, ovv = gmsh.model.occ.fragment([(3, 3)], [(3, 4)])
    ov, ovv = gmsh.model.occ.fragment([(2, 3)], [(2, 4)])

    print("fragment produced volumes:")
    for e in ov:
        print(e)

    # ovv contains the parent-child relationships for all the input entities:
    print("before/after fragment relations:")
    for e in ovv:
        print(e)

    gmsh.model.occ.synchronize()

    for e in ov:
        gmsh.model.add_physical_group(dim=3, tags=[e[1]])
        print(e)
    its = 1
    for surface in gmsh.model.getEntities(dim=2):
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        print(com)
        gmsh.model.addPhysicalGroup(2, [surface[1]], its)
        its += 1

    # for surface in gmsh.model.getEntities(dim=3):

    # Tag the left boundary
    # for line in gmsh.model.getEntities(dim=1):
    #    if np.isclose(com[0], 0) or np.isclose(com[0], 1) or np.isclose(com[1], 0) or  np.isclose(com[1], 1):
    gmsh.model.mesh.generate(gdim)

    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    mesh, cell_markers, facet_markers = gmshio.model_to_mesh(
        gmsh.model,
        mesh_comm,
        gmsh_model_rank,
        gdim=gdim,
    )

    mesh_hierarchy = []
    mesh_hierarchy.append(mesh)

    # def refine_all(x):
    for _i in range(n_ref):
        mesh.topology.create_entities(2)
        mesh.topology.create_entities(1)
        mesh = refine(mesh)
        mesh_hierarchy.append(mesh)
    return mesh_hierarchy
