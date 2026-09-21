from .. import _core
from .attention import EncAttention
from .module import EncModule


class EncBlock(EncModule):
    """GPT-2 transformer block; mirrors the C++ gpt2_block_ops step-for-step over the
    canonical loader weight names (ln_1/ln_2/up/down/...). Not a composable layer."""

    def __init__(self):
        super().__init__()
        self.attn = EncAttention()

    def forward(self, x):
        inf, fhe = self.inf, self.inf.fhe

        with inf.step("transformer_block"):
            with inf.step("block_in"):
                lvl = x.level + (1 if x.noise_deg == 2 else 0)
                inf.name_ct_if_absent(x, f"transformer_block.x-lvl={lvl}")
                skip = x

            with inf.step("ln_1"):
                normed = _core.norm(inf, x, "ln_1")
                if _core.fold_ln_affine("ln_1"):
                    with inf.step("ln_shift"):
                        inf.add_affine_term(normed, "ln_1.shift")
                    x = normed
                else:
                    x = _core.ln_affine(inf, normed, "ln_1")

            x = self.attn(x)
            with inf.step("attn_residual"):
                x = fhe.add(x, skip)
                skip = x

            with inf.step("ln_2"):
                x = _core.layer_norm(inf, x, "ln_2")

            with inf.step("up_linear"):
                inf.name_ct_if_absent(x, "mlp_block.x")
            with inf.step("up_linear"):
                up_hint = fhe.level_limit() - (2 if inf.complex else 1)
                fhe.bootstrap_hint(x, up_hint, True)
                fhe.level_hint(x, up_hint)
                if inf.complex:
                    x = _core.linear_outputpack(inf, x, "up", inf.size.hidDim, inf.size.expDim)
                else:
                    x = _core.linear(inf, x, "up", inf.size.hidDim, inf.size.expDim)
            with inf.step("gelu"):
                x = _core.gelu_approx(inf, x, "mlp.act")
            with inf.step("down_linear"):
                fhe.bootstrap_hint(x, fhe.level_limit() - 1, True)
                fhe.level_hint(x, fhe.level_limit() - 1)
                x = _core.linear(inf, x, "down", inf.size.expDim, inf.size.hidDim)

            with inf.step("mlp_residual"):
                x = fhe.add(x, skip)

        return x
