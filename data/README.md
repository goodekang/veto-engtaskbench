# Data

`splits/bench220.csv` lists 220 tasks. `features/<task_id>.npz` stores the observation bag written during sandboxed tool calls (entity or operation tokens, not the source media); the bags are distributed separately from this repository.

Public artefact sources (download separately if you want to re-extract):

- buildingSMART sample IFC models
- Schependomlaan open BIM dataset
- Fusion 360 Gallery reconstruction subset
- DeepCAD corpus

The 42-tool registry is `tools/registry.json`. Verifier oracles are code in `src/veto/verifier.py` and are not registered as agent-callable tools.

`aux/cases/` holds the two worked episodes used by `examples/infer_one_task.py` (door table extras live inside the corresponding feature archive).
