from .module import EncModule


class EncSequential(EncModule):
    def __init__(self, *modules):
        super().__init__()
        self._items = list(modules)
        for i, mod in enumerate(self._items):
            setattr(self, str(i), mod)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def forward(self, x):
        for mod in self._items:
            x = mod(x)
        return x


class EncModuleList(EncSequential):
    def forward(self, x):
        raise NotImplementedError("EncModuleList is a container; iterate it explicitly")
