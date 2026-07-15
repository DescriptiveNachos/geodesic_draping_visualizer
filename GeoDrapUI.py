"""Interactive GeoDrape viewer backed by the ``geodesic_draping`` bindings.

This is the bindings-based counterpart of ``Reference_GeoDrapeUI.py``.  Run it
from this directory so the reference mesh path resolves as expected::

    python GeoDrapUI.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geodesic_draping as gd
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import pyvista as pv


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MESH = APP_DIR / "meshes" / "DemoV5_s.stl"


def vf_from_pyvista(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous vertex and triangular-face arrays."""
    mesh = mesh.extract_surface().triangulate()
    faces = np.asarray(mesh.faces).reshape((-1, 4))[:, 1:]
    return (
        np.ascontiguousarray(mesh.points, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int64),
    )


def vf_to_pyvista(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    cells = np.column_stack((np.full(len(faces), 3), faces)).ravel()
    return pv.PolyData(vertices, cells)


def extract_boundary_loop(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return one ordered boundary loop, matching the reference outline use."""
    counts: dict[tuple[int, int], int] = {}
    directed: list[tuple[int, int]] = []
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            counts[key] = counts.get(key, 0) + 1
            directed.append((int(a), int(b)))

    adjacency: dict[int, list[int]] = {}
    for a, b in directed:
        if counts[(min(a, b), max(a, b))] == 1:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
    if not adjacency:
        return np.empty(0, dtype=np.int64)

    start = min(adjacency)
    loop = [start]
    previous, current = -1, start
    while True:
        choices = [item for item in adjacency[current] if item != previous]
        if not choices:
            break
        following = choices[0]
        if following == start:
            break
        if following in loop:
            break
        loop.append(following)
        previous, current = current, following
    return np.asarray(loop, dtype=np.int64)


def rotate_translate_b_to_a(
    points: np.ndarray, alpha: float, translation: tuple[float, float] = (0, 0)
) -> np.ndarray:
    alpha = np.radians(alpha)
    rotation = np.array(
        [[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]]
    )
    translated = np.asarray(points) @ rotation.T + np.asarray(translation)
    return np.column_stack((translated, np.zeros(len(translated))))


class ViewerController:
    def __init__(self, surface: pv.PolyData):
        self.vertices, self.faces = vf_from_pyvista(surface)
        self.params = {
            "x": 0.0,
            "y": 0.0,
            "angle": 0.0,
            "t_coeff": 1.0,
            "width": 350.0,
            "n_levels": 50,
            "toggle_contours": False,
            "toggle_outline": False,
        }
        self.solver: gd.GeoDrapeSolver
        self.result: gd.DrapeResult
        self.mesh = None
        self.origin_pc = None
        self.origin2d_pc = None
        self.dir_pc = None
        self.update_solver()
        self.solve()
        self._register_mesh()
        self._register_origins()
        self._register_directions()
        self.update_contours()
        self.update_outline()

    def update_solver(self) -> None:
        self.solver = gd.GeoDrapeSolver(
            self.vertices,
            self.faces,
            diffusion_time_coefficient=self.params["t_coeff"],
        )

    def solve(self) -> None:
        detailed = self.params["toggle_contours"] or self.params["toggle_outline"]
        self.result = self.solver.solve(
            np.array([self.params["x"], self.params["y"]]),
            self.params["angle"],
            mode="complete" if detailed else "fast",
            retrieval="extrinsic",
            sample_vertex_shear=True,
        )

    def _origin(self) -> np.ndarray:
        for family in self.result.generators:
            for line in family:
                points = np.asarray(line, dtype=float)
                if len(points):
                    return points[0]
        xy = np.array([self.params["x"], self.params["y"]])
        return self.vertices[np.argmin(np.linalg.norm(self.vertices[:, :2] - xy, axis=1))]

    def _direction_vectors(self) -> np.ndarray:
        vectors = []
        for family in self.result.generators:
            for line in family:
                points = np.asarray(line, dtype=float)
                if len(points) >= 2:
                    vector = points[1] - points[0]
                    norm = np.linalg.norm(vector)
                    if norm:
                        vectors.append(vector / norm)
        return np.asarray(vectors, dtype=float).reshape((-1, 3))

    def _register_mesh(self) -> None:
        self.mesh = ps.register_surface_mesh("mesh", self.vertices, self.faces)
        self._update_shear()

    def _update_shear(self) -> None:
        if self.result.vertex_shear is not None:
            self.mesh.add_scalar_quantity(
                "shear",
                self.result.vertex_shear,
                defined_on="vertices",
                enabled=True,
                cmap="jet",
            )

    def _register_origins(self) -> None:
        self.origin_pc = ps.register_point_cloud(
            "origin_on_surface", self._origin().reshape(1, 3), radius=0.005, color=(0, 1, 0)
        )
        self.origin2d_pc = ps.register_point_cloud(
            "origin_2D",
            np.array([[self.params["x"], self.params["y"], 0.0]]),
            radius=0.005,
            color=(0, 0, 1),
        )

    def _register_directions(self) -> None:
        vectors = self._direction_vectors()
        origins = np.tile(self._origin(), (len(vectors), 1))
        self.dir_pc = ps.register_point_cloud("direction_origins", origins, radius=0.0005)
        if len(vectors):
            self.dir_pc.add_vector_quantity("dir_vecs", vectors, enabled=True)

    @staticmethod
    def _pv_to_curve(poly: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
        edges, index = [], 0
        lines = np.asarray(poly.lines)
        while index < len(lines):
            count = int(lines[index])
            ids = lines[index + 1 : index + 1 + count]
            edges.extend((int(ids[j]), int(ids[j + 1])) for j in range(count - 1))
            index += count + 1
        return np.asarray(poly.points), np.asarray(edges, dtype=np.int64).reshape((-1, 2))

    def _compute_contours(self) -> tuple[pv.PolyData, pv.PolyData]:
        if self.result.distances is None:
            raise RuntimeError("Contours require a complete solve")
        width = self.params["width"]
        vmin, vmax = -width / 2, width / 2
        levels = np.linspace(vmin, vmax, self.params["n_levels"])
        surface = vf_to_pyvista(self.vertices, self.faces)
        surface.point_data["dist_0"] = self.result.distances[0]
        surface.point_data["dist_1"] = self.result.distances[1]
        cx = surface.threshold((vmin, vmax), scalars="dist_0").contour(levels, scalars="dist_1")
        cy = surface.threshold((vmin, vmax), scalars="dist_1").contour(levels, scalars="dist_0")
        return cx, cy

    def _register_contours(self, radius: float = 0.0005) -> None:
        for name in ("dist_1_isolines", "dist_0_isolines"):
            ps.remove_curve_network(name, error_if_absent=False)
        if not self.params["toggle_contours"]:
            return
        cx, cy = self._compute_contours()
        for name, poly, color in (
            ("dist_1_isolines", cx, (1, 0, 0)),
            ("dist_0_isolines", cy, (0, 1, 0)),
        ):
            points, edges = self._pv_to_curve(poly)
            if len(edges):
                ps.register_curve_network(name, points, edges, radius=radius, color=color)

    def _register_generators(self, radius: float = 0.001) -> None:
        for index in range(4):
            ps.remove_curve_network(f"gen{index}", error_if_absent=False)
            ps.remove_point_cloud(f"gen {index} points", error_if_absent=False)
        index = 0
        for family in self.result.generators:
            for line in family:
                points = np.asarray(line, dtype=float)
                if len(points):
                    ps.register_point_cloud(
                        f"gen {index} points", points, radius=radius, color=(1, 0, 0)
                    )
                if len(points) >= 2:
                    edges = np.column_stack((np.arange(len(points) - 1), np.arange(1, len(points))))
                    ps.register_curve_network(
                        f"gen{index}", points, edges, radius=radius, color=(1, 1, 1)
                    )
                index += 1

    def _register_outline(self, radius: float = 0.0005) -> None:
        ps.remove_curve_network("outline", error_if_absent=False)
        if not self.params["toggle_outline"]:
            return
        if self.result.distances is None:
            raise RuntimeError("Outline requires a complete solve")
        boundary = extract_boundary_loop(self.vertices, self.faces)
        if len(boundary) < 2:
            return
        d0, d1 = self.result.distances
        planar = np.column_stack((-d1[boundary], d0[boundary]))
        points = rotate_translate_b_to_a(
            planar,
            self.params["angle"],
            (self.params["x"], self.params["y"]),
        )
        edges = np.column_stack((np.arange(len(points)), np.roll(np.arange(len(points)), -1)))
        ps.register_curve_network("outline", points, edges, radius=radius, color=(1, 0, 0))

    def update_solution(self) -> None:
        self.solve()
        self._update_shear()
        origin = self._origin()
        self.origin_pc.update_point_positions(origin.reshape(1, 3))
        self.origin2d_pc.update_point_positions(
            np.array([[self.params["x"], self.params["y"], 0.0]])
        )
        vectors = self._direction_vectors()
        self.dir_pc.update_point_positions(np.tile(origin, (len(vectors), 1)))
        if len(vectors):
            self.dir_pc.add_vector_quantity("dir_vecs", vectors, enabled=True)

    def update_contours(self) -> None:
        self._register_contours()
        self._register_generators()

    def update_outline(self) -> None:
        self._register_outline()


def run(mesh_path: Path) -> None:
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Mesh not found: {mesh_path}. Pass an STL/OBJ/PLY path with --mesh."
        )
    surface = pv.read(mesh_path).extract_surface().triangulate()
    ps.init()
    controller = ViewerController(surface)

    def callback() -> None:
        xy_lim, angle_lim = 50, 45
        cx, controller.params["x"] = psim.SliderFloat("x", controller.params["x"], -xy_lim, xy_lim)
        cy, controller.params["y"] = psim.SliderFloat("y", controller.params["y"], -xy_lim, xy_lim)
        ca, controller.params["angle"] = psim.SliderFloat("angle", controller.params["angle"], -angle_lim, angle_lim)
        ct, controller.params["t_coeff"] = psim.SliderFloat("time coefficient", controller.params["t_coeff"], 0.01, 5)
        cw, controller.params["width"] = psim.SliderFloat("width", controller.params["width"], 10.0, 1000.0)
        cl, controller.params["n_levels"] = psim.SliderInt("levels", controller.params["n_levels"], 5, 200)
        cc, controller.params["toggle_contours"] = psim.Checkbox("Toggle Contours", controller.params["toggle_contours"])
        co, controller.params["toggle_outline"] = psim.Checkbox("Toggle Outline", controller.params["toggle_outline"])
        try:
            if ct:
                controller.update_solver()
            if cx or cy or ca or ct or cc or co:
                controller.update_solution()
                controller.update_contours()
                controller.update_outline()
            elif cw or cl:
                controller.update_contours()
        except Exception as error:
            print(f"Update failed -- keeping previous state: {error}")

    ps.set_user_callback(callback)
    ps.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    args = parser.parse_args()
    run(args.mesh.resolve())


if __name__ == "__main__":
    main()
