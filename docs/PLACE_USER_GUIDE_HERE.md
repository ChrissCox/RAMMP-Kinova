# Reference documents

Drop the **Kinova Gen3 Ultra lightweight robot User Guide** PDF in this `docs/` folder
(e.g. `User-Guide-Gen3-R07.pdf`). It is treated as a binary by `.gitattributes`.

Handy sections for this project:
- High-level vs. low-level control (`Kinova.Api.Base` vs `BaseCyclic` 1 kHz servoing).
- Joint position/speed/torque limits (7-DoF spherical wrist).
- Network interfaces & default IPs (EXT interface default `192.168.1.10`, `admin/admin`).
- Safety: no mechanical brakes; arm settles slowly on power loss; E-stop behavior.
- Predefined Home / Retract poses and the Kortex control modes.
