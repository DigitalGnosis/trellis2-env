"""Make TRELLIS.2's DINOv3 feature extractor work on transformers >= 5.4.

Upstream indexes `self.model.layer`; transformers 5.4+ moved the layers (5.16.1 keeps
them at `model.model.layer`; PR #148's `.encoder.layer` is also wrong). Locate the first
ModuleList instead of naming the path, and tolerate tuple outputs. Proven on pop-os CPU
with the real gated weights (2026-09-01) and on the A40 (runs 5 and 6).
"""
import pathlib
import sys

OLD = """        for i, layer_module in enumerate(self.model.layer):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )
"""
NEW = """        _layers = next(m for _, m in self.model.named_modules() if isinstance(m, torch.nn.ModuleList))
        for i, layer_module in enumerate(_layers):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )
            if isinstance(hidden_states, (tuple, list)):
                hidden_states = hidden_states[0]
"""

path = pathlib.Path(sys.argv[1])
text = path.read_text()
if NEW in text:
    print("DINOV3_PATCH=ALREADY_APPLIED")
    sys.exit(0)
if OLD not in text:
    print("DINOV3_PATCH=LOOP_NOT_FOUND", file=sys.stderr)
    sys.exit(1)
path.write_text(text.replace(OLD, NEW, 1))
print("DINOV3_PATCH=APPLIED")
