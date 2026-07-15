# geodesic_draping_visualizer

An interactive visualization tool for fabric draping based on
`geodesic_draping`, Polyscope, and PyVista.

## Run

From this directory, using the environment in which `geodesic_draping` is
installed:

```powershell
python GeoDrapUI.py
```

The default mesh is `meshes/DemoV5_s.stl`. Select another mesh with:

```powershell
python GeoDrapUI.py --mesh path\to\mesh.stl
```

## Controls

- Choose `fast`, `hybrid`, or `complete` solve mode explicitly.
- Solves update continuously by default. Enable **Manual solve** to apply seed,
  angle, and mode changes only when **Solve** is pressed.
- Choose the `signpost` or `integer` intrinsic backend and the `none`, `flip`,
  or `refine` refinement mode under solver settings. These constructor settings
  take effect together when **Apply solver settings** is pressed.
- Contours and the planar outline require a mode which returns distances.
- Surface direction fields are displayed on the internal subdivision mesh.
- Use Polyscope's normal sidebar to control scalar/vector quantity visibility,
  styling, ranges, and vector sizes.
