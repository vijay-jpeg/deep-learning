from Value import Value

class Vec:
    def __init__(self, data) -> None:
        self.data = []

        for s in data:
            if not isinstance(s, Value): s = Value(s)
            self.data.append(s)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        return self.data[i]

    def __iter__(self):
        for v in self.data: yield v

    def __repr__(self):
        return f"Vec({[v.data for v in self.data]})"

    def __add__(self, other):
        new_data = []
        if isinstance(other, Vec): 
            assert len(self) == len(other)
            for i in range (len(self)):
                new_data.append(self[i] + other[i])
        else:
            for i in range (len(self)):
                new_data.append(self[i] + other)

        return Vec(new_data)

    def __sub__(self, other):
        new_data = []
        if isinstance(other, Vec): 
            assert len(self) == len(other)
            for i in range (len(self)):
                new_data.append(self[i] - other[i])
        else:
            for i in range (len(self)):
                new_data.append(self[i] - other)

        return Vec(new_data)

    def __mul__(self, other):
        new_data = []
        if isinstance(other, Vec): 
            assert len(self) == len(other)
            for i in range (len(self)):
                new_data.append(self[i] * other[i])
        else:
            for i in range (len(self)):
                new_data.append(self[i] * other)

        return Vec(new_data)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return -1*self + other

    def dot(self, other) -> Value:
        out = Value(0)
        if isinstance(other, Vec): 
            assert len(self) == len(other)
            for i in range (len(self)):
                out += self[i] * other[i]
        else:
            for i in range (len(self)):
                out += self[i] * other

        return out

    def sum(self) -> Value:
        out = Value(0)
        for v in self:
            out += v
        return out

    def relu(self):
        return Vec([v.relu() for v in self])