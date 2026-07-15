"""Interactive GeoDrape viewer backed by the ``geodesic_draping`` bindings."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import traceback

import geodesic_draping as gd
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import pyvista as pv


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MESH = APP_DIR / "meshes" / "DemoV5_s.stl"
MODES = ("fast", "hybrid", "complete")
BACKENDS = ("signpost", "integer")
REFINEMENTS = ("none", "flip", "refine")


def vf_from_pyvista(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous vertex and triangular-face arrays."""
    triangular = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    faces = np.asarray(triangular.faces).reshape((-1, 4))[:, 1:]
    return (
        np.ascontiguousarray(triangular.points, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int64),
    )


def vf_to_pyvista(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    cells = np.column_stack((np.full(len(faces), 3), faces)).ravel()
    return pv.PolyData(vertices, cells)


def extract_boundary_loop(faces: np.ndarray) -> np.ndarray:
    """Return one ordered boundary loop."""
    edge_counts: dict[tuple[int, int], int] = {}
    directed_edges: list[tuple[int, int]] = []
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            a, b = int(a), int(b)
            key = (min(a, b), max(a, b))
            edge_counts[key] = edge_counts.get(key, 0) + 1
            directed_edges.append((a, b))

    adjacency: dict[int, list[int]] = {}
    for a, b in directed_edges:
        if edge_counts[(min(a, b), max(a, b))] == 1:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
    if not adjacency:
        return np.empty(0, dtype=np.int64)

    start = min(adjacency)
    loop = [start]
    previous, current = -1, start
    while True:
        choices = [vertex for vertex in adjacency[current] if vertex != previous]
        if not choices:
            break
        following = choices[0]
        if following == start or following in loop:
            break
        loop.append(following)
        previous, current = current, following
    return np.asarray(loop, dtype=np.int64)


def rotate_translate_b_to_a(
    points: np.ndarray,
    angle_degrees: float,
    translation: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    angle = np.radians(angle_degrees)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    translated = np.asarray(points) @ rotation.T + np.asarray(translation)
    return np.column_stack((translated, np.zeros(len(translated))))


class DrapeSession:
    """Own the mesh, persistent solver, and results for the latest solve."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        intrinsic_backend: str = "signpost",
        refinement: str = "none",
        diffusion_time_coefficient: float = 1.0,
    ) -> None:
        self.vertices = vertices
        self.faces = faces
        self.intrinsic_backend = intrinsic_backend
        self.refinement = refinement
        self.diffusion_time_coefficient = diffusion_time_coefficient
        self.solver: gd.GeoDrapeSolver
        self.result: gd.DrapeResult | None = None
        self.subdivision_result: gd.DrapeResult | None = None
        self.last_solve_seconds: float | None = None
        self._build_solver()

    def _build_solver(self) -> None:
        self.solver = gd.GeoDrapeSolver(
            self.vertices,
            self.faces,
            intrinsic_backend=self.intrinsic_backend,
            refinement=self.refinement,
            diffusion_time_coefficient=self.diffusion_time_coefficient,
        )

    def apply_solver_settings(
        self,
        *,
        intrinsic_backend: str,
        refinement: str,
        diffusion_time_coefficient: float,
        seed_xy: np.ndarray,
        fabric_angle: float,
        mode: str,
    ) -> tuple[gd.DrapeResult, gd.DrapeResult]:
        """Build and solve with a replacement, committing only on success."""
        replacement = gd.GeoDrapeSolver(
            self.vertices,
            self.faces,
            intrinsic_backend=intrinsic_backend,
            refinement=refinement,
            diffusion_time_coefficient=diffusion_time_coefficient,
        )
        started = perf_counter()
        result = replacement.solve(
            seed_xy,
            fabric_angle,
            mode=mode,
            retrieval="extrinsic",
            sample_vertex_shear=True,
        )
        elapsed = perf_counter() - started
        subdivision = replacement.retrieve(retrieval="subdivision")
        self.solver = replacement
        self.intrinsic_backend = intrinsic_backend
        self.refinement = refinement
        self.diffusion_time_coefficient = diffusion_time_coefficient
        self.result = result
        self.subdivision_result = subdivision
        self.last_solve_seconds = elapsed
        return result, subdivision

    def solve(self, seed_xy: np.ndarray, fabric_angle: float, mode: str) -> gd.DrapeResult:
        started = perf_counter()
        result = self.solver.solve(
            seed_xy,
            fabric_angle,
            mode=mode,
            retrieval="extrinsic",
            sample_vertex_shear=True,
        )
        elapsed = perf_counter() - started
        self.result = result
        self.subdivision_result = None
        self.last_solve_seconds = elapsed
        return result

    def retrieve_subdivision(self) -> gd.DrapeResult:
        if self.result is None:
            raise RuntimeError("Run a solve before retrieving subdivision data")
        if self.subdivision_result is None:
            self.subdivision_result = self.solver.retrieve(retrieval="subdivision")
        return self.subdivision_result


class DrapeView:
    """Own Polyscope structures and derived visualization geometry."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.vertices = vertices
        self.faces = faces
        self.mesh = ps.register_surface_mesh("mesh", vertices, faces)
        self.subdivision_mesh = None
        self.origin_pc = None
        self.origin2d_pc = None
        self.direction_pc = None
        self.generator_count = 0
        self._original_quantities: set[str] = set()
        self._subdivision_quantities: set[str] = set()
        self._boundary = extract_boundary_loop(faces)

    @staticmethod
    def origin(result: gd.DrapeResult, seed_xy: np.ndarray) -> np.ndarray:
        for family in result.generators:
            for line in family:
                points = np.asarray(line, dtype=float)
                if len(points):
                    return points[0]
        distances = np.linalg.norm(result.vertices[:, :2] - seed_xy, axis=1)
        return result.vertices[np.argmin(distances)]

    @staticmethod
    def seed_directions(result: gd.DrapeResult) -> np.ndarray:
        vectors = []
        for family in result.generators:
            for line in family:
                points = np.asarray(line, dtype=float)
                if len(points) >= 2:
                    vector = points[1] - points[0]
                    norm = np.linalg.norm(vector)
                    if norm:
                        vectors.append(vector / norm)
        return np.asarray(vectors, dtype=float).reshape((-1, 3))

    @staticmethod
    def _remove_quantity(structure, names: set[str], name: str) -> None:
        if name in names:
            structure.remove_quantity(name, error_if_absent=False)
            names.discard(name)

    def update_seed_input(self, seed_xy: np.ndarray) -> None:
        position = np.array([[seed_xy[0], seed_xy[1], 0.0]])
        if self.origin2d_pc is None:
            self.origin2d_pc = ps.register_point_cloud(
                "origin_2D", position, radius=0.005, color=(0, 0, 1)
            )
        else:
            self.origin2d_pc.update_point_positions(position)

    def update_solution(
        self,
        result: gd.DrapeResult,
        subdivision: gd.DrapeResult,
        seed_xy: np.ndarray,
        *,
        surface_fields_enabled: bool,
    ) -> None:
        self._update_original_quantities(result)
        self._update_subdivision(subdivision, surface_fields_enabled)
        self._update_origins_and_seed_directions(result, seed_xy)
        self._update_generators(result)

    def _update_original_quantities(self, result: gd.DrapeResult) -> None:
        quantities = {
            "vertex shear": result.vertex_shear,
            "distance 0": None if result.distances is None else result.distances[0],
            "distance 1": None if result.distances is None else result.distances[1],
        }
        for name, values in quantities.items():
            if values is None:
                self._remove_quantity(self.mesh, self._original_quantities, name)
                continue
            first_registration = name not in self._original_quantities
            self.mesh.add_scalar_quantity(
                name,
                values,
                defined_on="vertices",
                enabled=(name == "vertex shear") if first_registration else None,
                cmap="jet" if name == "vertex shear" else None,
            )
            self._original_quantities.add(name)

    def _update_subdivision(
        self, result: gd.DrapeResult, surface_fields_enabled: bool
    ) -> None:
        self.subdivision_mesh = ps.register_surface_mesh(
            "subdivision mesh", result.vertices, result.faces, enabled=True
        )
        if result.face_shear is not None:
            first_registration = "face shear" not in self._subdivision_quantities
            self.subdivision_mesh.add_scalar_quantity(
                "face shear",
                result.face_shear,
                defined_on="faces",
                enabled=False if first_registration else None,
                cmap="jet",
            )
            self._subdivision_quantities.add("face shear")
        else:
            self._remove_quantity(
                self.subdivision_mesh, self._subdivision_quantities, "face shear"
            )

        if result.direction_fields is None:
            for name in ("direction field 0", "direction field 1"):
                self._remove_quantity(
                    self.subdivision_mesh, self._subdivision_quantities, name
                )
            return
        for family in range(2):
            name = f"direction field {family}"
            self.subdivision_mesh.add_vector_quantity(
                name,
                result.direction_fields[family],
                defined_on="faces",
                enabled=surface_fields_enabled,
            )
            self._subdivision_quantities.add(name)

    def set_surface_fields_enabled(
        self, result: gd.DrapeResult, enabled: bool
    ) -> None:
        if self.subdivision_mesh is None or result.direction_fields is None:
            return
        for family in range(2):
            name = f"direction field {family}"
            self.subdivision_mesh.add_vector_quantity(
                name,
                result.direction_fields[family],
                defined_on="faces",
                enabled=enabled,
            )
            self._subdivision_quantities.add(name)

    def _update_origins_and_seed_directions(
        self, result: gd.DrapeResult, seed_xy: np.ndarray
    ) -> None:
        origin = self.origin(result, seed_xy)
        if self.origin_pc is None:
            self.origin_pc = ps.register_point_cloud(
                "origin_on_surface",
                origin.reshape(1, 3),
                radius=0.005,
                color=(0, 1, 0),
            )
        else:
            self.origin_pc.update_point_positions(origin.reshape(1, 3))
        self.update_seed_input(seed_xy)

        vectors = self.seed_directions(result)
        bases = np.tile(origin, (len(vectors), 1))
        if self.direction_pc is None:
            self.direction_pc = ps.register_point_cloud(
                "direction_origins", bases, radius=0.0005
            )
        else:
            self.direction_pc.update_point_positions(bases)
        if len(vectors):
            self.direction_pc.add_vector_quantity("dir_vecs", vectors, enabled=None)
        else:
            self.direction_pc.remove_quantity("dir_vecs", error_if_absent=False)

    def _update_generators(self, result: gd.DrapeResult, radius: float = 0.001) -> None:
        lines = [np.asarray(line, dtype=float) for family in result.generators for line in family]
        for index in range(max(self.generator_count, len(lines))):
            ps.remove_curve_network(f"generator {index}", error_if_absent=False)
            ps.remove_point_cloud(f"generator {index} points", error_if_absent=False)
        for index, points in enumerate(lines):
            if len(points):
                ps.register_point_cloud(
                    f"generator {index} points",
                    points,
                    radius=radius,
                    color=(1, 0, 0),
                )
            if len(points) >= 2:
                edges = np.column_stack(
                    (np.arange(len(points) - 1), np.arange(1, len(points)))
                )
                ps.register_curve_network(
                    f"generator {index}",
                    points,
                    edges,
                    radius=radius,
                    color=(1, 1, 1),
                )
        self.generator_count = len(lines)

    @staticmethod
    def _pv_to_curve(poly: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
        edges, index = [], 0
        lines = np.asarray(poly.lines)
        while index < len(lines):
            count = int(lines[index])
            ids = lines[index + 1 : index + 1 + count]
            edges.extend((int(ids[j]), int(ids[j + 1])) for j in range(count - 1))
            index += count + 1
        return (
            np.asarray(poly.points),
            np.asarray(edges, dtype=np.int64).reshape((-1, 2)),
        )

    def update_contours(
        self,
        result: gd.DrapeResult,
        *,
        enabled: bool,
        width: float,
        level_count: int,
        radius: float = 0.0005,
    ) -> None:
        for name in ("distance 1 isolines", "distance 0 isolines"):
            ps.remove_curve_network(name, error_if_absent=False)
        if not enabled:
            return
        if result.distances is None:
            raise RuntimeError("Contours require hybrid or complete mode")

        vmin, vmax = -width / 2, width / 2
        levels = np.linspace(vmin, vmax, level_count)
        surface = vf_to_pyvista(self.vertices, self.faces)
        surface.point_data["distance 0"] = result.distances[0]
        surface.point_data["distance 1"] = result.distances[1]
        contours = (
            (
                "distance 1 isolines",
                surface.threshold((vmin, vmax), scalars="distance 0").contour(
                    levels, scalars="distance 1"
                ),
                (1, 0, 0),
            ),
            (
                "distance 0 isolines",
                surface.threshold((vmin, vmax), scalars="distance 1").contour(
                    levels, scalars="distance 0"
                ),
                (0, 1, 0),
            ),
        )
        for name, polyline, color in contours:
            points, edges = self._pv_to_curve(polyline)
            if len(edges):
                ps.register_curve_network(
                    name, points, edges, radius=radius, color=color
                )

    def update_outline(
        self,
        result: gd.DrapeResult,
        *,
        enabled: bool,
        seed_xy: np.ndarray,
        fabric_angle: float,
        radius: float = 0.0005,
    ) -> None:
        ps.remove_curve_network("outline", error_if_absent=False)
        if not enabled:
            return
        if result.distances is None:
            raise RuntimeError("Outline requires hybrid or complete mode")
        if len(self._boundary) < 2:
            return
        d0, d1 = result.distances
        planar = np.column_stack((-d1[self._boundary], d0[self._boundary]))
        points = rotate_translate_b_to_a(
            planar, fabric_angle, (float(seed_xy[0]), float(seed_xy[1]))
        )
        edges = np.column_stack(
            (np.arange(len(points)), np.roll(np.arange(len(points)), -1))
        )
        ps.register_curve_network(
            "outline", points, edges, radius=radius, color=(1, 0, 0)
        )

    @staticmethod
    def clear_contours_and_outline() -> None:
        for name in ("distance 1 isolines", "distance 0 isolines", "outline"):
            ps.remove_curve_network(name, error_if_absent=False)


class ViewerController:
    """Own UI state and coordinate the session and view."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.params = {
            "x": 0.0,
            "y": 0.0,
            "angle": 0.0,
            "mode": "fast",
            "manual_solve": False,
            "width": 350.0,
            "n_levels": 50,
            "show_contours": False,
            "show_outline": False,
            "show_surface_fields": False,
        }
        self.pending_settings = {
            "intrinsic_backend": "signpost",
            "refinement": "none",
            "diffusion_time_coefficient": 1.0,
        }
        self.session = DrapeSession(vertices, faces)
        self.view = DrapeView(vertices, faces)
        self.error_message: str | None = None
        self.solve()

    @property
    def seed_xy(self) -> np.ndarray:
        return np.array([self.params["x"], self.params["y"]], dtype=float)

    def _run_guarded(self, operation: str, callback) -> bool:
        try:
            callback()
        except Exception as error:
            self.error_message = f"{operation} failed: {error}"
            print(self.error_message)
            traceback.print_exc()
            return False
        self.error_message = None
        return True

    def solve(self) -> None:
        def operation() -> None:
            result = self.session.solve(
                self.seed_xy, self.params["angle"], self.params["mode"]
            )
            subdivision = self.session.retrieve_subdivision()
            self.view.update_solution(
                result,
                subdivision,
                self.seed_xy,
                surface_fields_enabled=self.params["show_surface_fields"],
            )
            self._update_distance_geometry(result)

        self._run_guarded("Solve", operation)

    def apply_solver_settings(self) -> None:
        coefficient = self.pending_settings["diffusion_time_coefficient"]
        if not np.isfinite(coefficient) or coefficient <= 0:
            self.error_message = "Solver settings failed: time coefficient must be positive"
            return

        def operation() -> None:
            result, subdivision = self.session.apply_solver_settings(
                **self.pending_settings,
                seed_xy=self.seed_xy,
                fabric_angle=self.params["angle"],
                mode=self.params["mode"],
            )
            self.view.update_solution(
                result,
                subdivision,
                self.seed_xy,
                surface_fields_enabled=self.params["show_surface_fields"],
            )
            self._update_distance_geometry(result)

        self._run_guarded("Solver settings", operation)

    def _update_distance_geometry(self, result: gd.DrapeResult) -> None:
        if result.distances is None:
            self.view.clear_contours_and_outline()
            return
        self.view.update_contours(
            result,
            enabled=self.params["show_contours"],
            width=self.params["width"],
            level_count=self.params["n_levels"],
        )
        self.view.update_outline(
            result,
            enabled=self.params["show_outline"],
            seed_xy=self.seed_xy,
            fabric_angle=self.params["angle"],
        )

    def update_contours(self) -> None:
        if self.session.result is None:
            return
        self._run_guarded(
            "Contour update",
            lambda: self.view.update_contours(
                self.session.result,
                enabled=self.params["show_contours"],
                width=self.params["width"],
                level_count=self.params["n_levels"],
            ),
        )

    def update_outline(self) -> None:
        if self.session.result is None:
            return
        self._run_guarded(
            "Outline update",
            lambda: self.view.update_outline(
                self.session.result,
                enabled=self.params["show_outline"],
                seed_xy=self.seed_xy,
                fabric_angle=self.params["angle"],
            ),
        )

    def update_surface_fields(self) -> None:
        subdivision = self.session.subdivision_result
        if subdivision is None:
            return
        self._run_guarded(
            "Direction-field update",
            lambda: self.view.set_surface_fields_enabled(
                subdivision, self.params["show_surface_fields"]
            ),
        )

    def callback(self) -> None:
        solve_inputs_changed = self._draw_solve_controls()
        settings_applied = self._draw_solver_settings()
        display_changed = self._draw_display_controls()
        self._draw_status()

        self.view.update_seed_input(self.seed_xy)
        if settings_applied:
            self.apply_solver_settings()
        elif solve_inputs_changed and not self.params["manual_solve"]:
            self.solve()
        if display_changed["contours"]:
            self.update_contours()
        if display_changed["outline"]:
            self.update_outline()
        if display_changed["surface_fields"]:
            self.update_surface_fields()

    def _draw_solve_controls(self) -> bool:
        changed = False
        item_changed, self.params["x"] = psim.SliderFloat(
            "x", self.params["x"], -50.0, 50.0
        )
        changed |= item_changed
        item_changed, self.params["y"] = psim.SliderFloat(
            "y", self.params["y"], -50.0, 50.0
        )
        changed |= item_changed
        item_changed, self.params["angle"] = psim.SliderFloat(
            "angle", self.params["angle"], -45.0, 45.0
        )
        changed |= item_changed
        mode_index = MODES.index(self.params["mode"])
        item_changed, mode_index = psim.Combo("solve mode", mode_index, MODES)
        if item_changed:
            self.params["mode"] = MODES[mode_index]
        changed |= item_changed
        _, self.params["manual_solve"] = psim.Checkbox(
            "Manual solve", self.params["manual_solve"]
        )
        if psim.Button("Solve"):
            self.solve()
        return changed

    def _draw_solver_settings(self) -> bool:
        psim.SeparatorText("Solver settings")
        backend_index = BACKENDS.index(self.pending_settings["intrinsic_backend"])
        changed, backend_index = psim.Combo(
            "intrinsic backend", backend_index, BACKENDS
        )
        if changed:
            self.pending_settings["intrinsic_backend"] = BACKENDS[backend_index]
        refinement_index = REFINEMENTS.index(self.pending_settings["refinement"])
        changed, refinement_index = psim.Combo(
            "refinement", refinement_index, REFINEMENTS
        )
        if changed:
            self.pending_settings["refinement"] = REFINEMENTS[refinement_index]
        _, coefficient = psim.InputFloat(
            "time coefficient",
            self.pending_settings["diffusion_time_coefficient"],
            0.0,
            0.0,
            "%.4f",
        )
        self.pending_settings["diffusion_time_coefficient"] = coefficient
        return psim.Button("Apply solver settings")

    def _draw_display_controls(self) -> dict[str, bool]:
        psim.SeparatorText("Derived geometry")
        has_distances = (
            self.session.result is not None
            and self.session.result.distances is not None
        )
        psim.BeginDisabled(not has_distances)
        contours_changed, self.params["show_contours"] = psim.Checkbox(
            "Toggle Contours", self.params["show_contours"]
        )
        outline_changed, self.params["show_outline"] = psim.Checkbox(
            "Toggle Outline", self.params["show_outline"]
        )
        width_changed, self.params["width"] = psim.SliderFloat(
            "width", self.params["width"], 10.0, 1000.0
        )
        levels_changed, self.params["n_levels"] = psim.SliderInt(
            "levels", self.params["n_levels"], 5, 200
        )
        psim.EndDisabled()
        if not has_distances:
            psim.TextDisabled("Contours and outline require hybrid or complete mode")

        fields_available = (
            self.session.subdivision_result is not None
            and self.session.subdivision_result.direction_fields is not None
        )
        psim.BeginDisabled(not fields_available)
        fields_changed, self.params["show_surface_fields"] = psim.Checkbox(
            "Surface direction fields", self.params["show_surface_fields"]
        )
        psim.EndDisabled()
        if not fields_available:
            psim.TextDisabled("Surface direction fields are unavailable in this result")
        return {
            "contours": contours_changed or width_changed or levels_changed,
            "outline": outline_changed,
            "surface_fields": fields_changed,
        }

    def _draw_status(self) -> None:
        if self.error_message:
            psim.Separator()
            psim.TextColored((1.0, 0.25, 0.25, 1.0), self.error_message)
            if psim.SmallButton("Clear error"):
                self.error_message = None
        if psim.CollapsingHeader("Diagnostics"):
            elapsed = self.session.last_solve_seconds
            psim.Text("Last solve: --" if elapsed is None else f"Last solve: {elapsed * 1000:.2f} ms")
            psim.Text(
                f"Original mesh: {len(self.session.vertices)} vertices, "
                f"{len(self.session.faces)} faces"
            )
            subdivision = self.session.subdivision_result
            if subdivision is not None:
                psim.Text(
                    f"Subdivision: {len(subdivision.vertices)} vertices, "
                    f"{len(subdivision.faces)} faces"
                )


def run(mesh_path: Path) -> None:
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Mesh not found: {mesh_path}. Pass an STL/OBJ/PLY path with --mesh."
        )
    vertices, faces = vf_from_pyvista(pv.read(mesh_path))
    ps.init()
    controller = ViewerController(vertices, faces)
    ps.set_user_callback(controller.callback)
    ps.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    args = parser.parse_args()
    run(args.mesh.resolve())


if __name__ == "__main__":
    main()
