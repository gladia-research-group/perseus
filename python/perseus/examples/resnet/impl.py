"""ResNet-20 on the impl driver: `ResnetImpl` runs the network's stages (model.ResNet20.stages) as capture / plan
stages on a session (the CPU twin for the capture), `ResnetPlanned` records them once and runs the compiled programs on
the device (`EncoderPlannedPrefill` with no PCMM linears: the convs are plaintext diagonals)."""
from __future__ import annotations

import numpy as np

from perseus.impl.driver import ImplModel
from perseus.examples.bert.planned import EncoderPlannedPrefill
from perseus.impl.prefill import name_inputs


class ResnetImpl(ImplModel):
    def __init__(self, inf, net, *, core=None, profile=False):
        super().__init__(inf, core=core, profile=profile)
        self.net = net
        self.n_layers = len(net.stages())

    def prefill_stages(self, ch=None):
        def stage(b, lab, fn):
            def compute(xs):
                cap = self.enter_stage(b, 0)
                name_inputs(self.inf, xs)
                with self.rt.step(lab):
                    xs = fn(self.rt.ops, xs)
                self.exit_stage(b, cap)
                return xs
            return compute
        return [(lab, None, stage(b, lab, fn)) for b, (lab, fn) in enumerate(self.net.stages())]

    def encode(self, img):
        """The image as fresh ciphertexts (an encrypted zero plus its slot vectors)."""
        ops = self.rt.ops
        out = []
        for v in self.net.encode(img):
            zero = ops.core.encode_token_input(ops.inf, np.zeros(self.rt.dims.dim))
            out.append(ops.add_pt(zero, v))
        return out

    def prefill(self, img, on_block=None):
        self.start()
        self.token_begin(0)
        xs = self.run_stages(self.encode(img), 0, self.prefill_stages(None))
        return xs, None, self.token_end(0)

    def logits(self, xs):
        return self.net.logits([np.real(self.rt.ops.decrypt_slots_complex(x)) for x in xs])


class ResnetPlanned(EncoderPlannedPrefill):
    """Each stage compiled from the previous stage's compiled exit (EncoderPlannedPrefill.prepare). `tfhe_chunk` > 0:
    the session serves the TFHE ReLU's LUTs (impl/tfhe.py), `tfhe_chunk` values per switch pass."""
    tfhe_chunk = 0

    def linears(self):
        return {}

    def session_kw(self):
        return {"tfhe": self.tfhe_chunk} if self.tfhe_chunk else {}

    def after_setup(self):
        if self.tfhe_chunk:
            from perseus.impl import tfhe
            tfhe.setup_session(self.sess)

    def input_values(self, img):
        return self.model.net.encode(img)

    def logits(self, regs):
        return self.model.net.logits([np.real(v) for v in self.decrypt(regs)])
