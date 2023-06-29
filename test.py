import numpy as np
x = np.zeros((2))
class X:
    def __init__(self, x) -> None:
        self.x = np.array(x)
x1 = X(x)
x[0] = 1
print(x1.x)
print(x)