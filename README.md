# geodesic_draping_visualizer

Interactive visualization tool for fabric draping based on `geodesic_draping`, Polyscope, and PyVista. Includes a demo STL mesh.

## Install

This demo was tested with Python 3.11.

```powershell
python -m pip install geodesic-draping numpy pyvista polyscope
```

Optional `uv` setup:

```powershell
uv venv
uv pip install geodesic-draping numpy pyvista polyscope
```

## Run

```powershell
python GeoDrapUI.py
```

By default this loads `meshes/DemoV5_s.stl`.

With another mesh:

```powershell
python GeoDrapUI.py --mesh path\to\mesh.stl
```

## Controls

- Choose `fast`, `hybrid`, or `complete` solve mode explicitly.
- Solves update continuously by default. Enable **Manual solve** to apply seed, angle, and mode changes only when **Solve** is pressed.
- Choose the `signpost` or `integer` intrinsic backend and the `none`, `flip`, or `refine` refinement mode under solver settings. These constructor settings take effect together when **Apply solver settings** is pressed.
- Contours and the planar outline require a mode which returns distances.
- Surface direction fields are displayed on the internal subdivision mesh.
- Use Polyscope's normal sidebar to control scalar/vector quantity visibility, styling, ranges, and vector sizes.
