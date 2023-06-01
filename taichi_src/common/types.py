import numpy as np
import taichi as ti
from enum import Enum

double = ti.types.f32

vec1 = ti.types.vector(n=1, dtype=double)
vec2 = ti.types.vector(n=2, dtype=double)
vec3 = ti.types.vector(n=3, dtype=double)
vec4 = ti.types.vector(n=4, dtype=double)
vec5 = ti.types.vector(n=5, dtype=double)
mat3x3 = ti.types.matrix(n=3, m=3, dtype=double)

vec0i = ti.types.vector(n=0, dtype=int)
vec1i = ti.types.vector(n=1, dtype=int)
vec2i = ti.types.vector(n=2, dtype=int)
vec3i = ti.types.vector(n=3, dtype=int)

mat3x3i = ti.types.matrix(n=3, m=3, dtype=int)
mat3x2i = ti.types.matrix(3, 2, int)
mat5x2i = ti.types.matrix(5, 2, int)
mat5x3i = ti.types.matrix(5, 3, int)

kron = mat3x3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
kroni = mat3x3i([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

levi_chevita = np.array([
    [[0, 0, 0],
     [0, 0, 1],
     [0, -1, 0]],
    [[0, 0, -1],
     [0, 0, 0],
     [1, 0, 0]],
    [[0, 1, 0],
     [-1, 0, 0],
     [0, 0, 0]],
], dtype=np.float64)

class NonHallLES(Enum):
    DNS = "DNS"
    SMAG = "SMAGORINSKY"
    CROSS_HELICITY = "CROSS_HELICITY"

class Initials(Enum):
    OT = "OT"
    RAND = "RAND"
