class EncModule:
    """Base for encrypted modules; children register on attribute assignment."""

    def __init__(self):
        object.__setattr__(self, "_children", {})
        object.__setattr__(self, "inf", None)

    def __setattr__(self, name, value):
        if isinstance(value, EncModule):
            self._children[name] = value
        object.__setattr__(self, name, value)

    def bind(self, inf):
        object.__setattr__(self, "inf", inf)
        for child in self._children.values():
            child.bind(inf)
        return self

    def children(self):
        return self._children.values()

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)
