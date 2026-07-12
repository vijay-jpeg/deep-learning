from Value import Value
from Vec import Vec

class Mat:
    """A 2-D matrix of `Value` scalars (row-major `List[List[Value]]`). Ops compose
    scalar `Value` arithmetic; no `backward()` (reduce to a scalar first). Each op
    returns a NEW `Mat`."""

    def __init__(self, data) -> None:
        self.data = []
        self.rows = len(data)
        self.cols = len(data[0])
        
        for row in data:
            assert len(row) == self.cols
            new_row = []
            for v in row:
                if not isinstance(v, Value): v = Value(v)
                new_row.append(v)
            self.data.append(new_row)
    
    @property
    def shape(self):
        return (self.rows, self.cols)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __iter__(self):
        for v in self.data: yield v

    def __repr__(self):
        return f"Mat(shape={self.shape})"
    
    def get_col(self, i):
        out = []
        for row in self:
            out.append(row[i])

        return out

    def matmul(self, other):
        assert self.cols == other.rows

        out = Mat([[] for _ in range(self.rows)])

        for i in range(self.rows):
            for j in range(other.cols):
                dot = Vec(self[i]).dot(other.get_col(j))
                out[i].append(dot)

        out.cols = other.cols
        return out




    def __matmul__(self, other):
        return self.matmul(other)
    
    def transpose(self):
        tp = Mat([[] for _ in self[0]])

        for j in range(len(self[0])):
            for i in range(len(self)):
                tp[j].append(self[i][j])

        tp.rows = len(tp)
        tp.cols = len(tp[0])

        return tp

    @property
    def T(self):
        return self.transpose()

    def reshape(self, rows, cols):
        if (self.rows * self.cols != rows*cols): raise ValueError

        values = []
        for i in range(len(self)):
            for j in range(len(self[0])):
                values.append(self[i][j])
        
        rs = []

        for r in range(rows):
            rs.append(values[r * cols: (r+1) * cols])
        
        return Mat(rs)

    def sum(self):
        out = Value(0)
        for row in self:
            for v in row:
                out += v

        return out

    def mean(self):
        total = self.sum()
        count = len(self) * len(self[0])

        return total / count