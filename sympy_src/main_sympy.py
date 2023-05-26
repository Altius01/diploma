import sympy as sym

from sympy import Matrix, Symbol, Float

class ConstantSymbol(sym.NumberSymbol):
    __slots__ = ['name', 'value']

    def __new__(cls, name, value):
        self = super(ConstantSymbol,cls).__new__(cls)
        self.name  = name
        self.value = Float(value)
        return self

    def _as_mpf_val(self, prec):
        return self.value._as_mpf_val(prec)

    def _sympystr(self, printer):
        return printer.doprint(Symbol(self.name))

    def __eq__(self, rhs):
        return (isinstance(rhs, ConstantSymbol)
                and (self.name  == rhs.name)
                and (self.value == rhs.value))
    def __hash__(self) :
        return 113*hash(self.name) + 29*hash(self.value)

def tesnsordot(a: Matrix, b: Matrix):
    if not 1 in a.shape or not 1 in b.shape:
        Print("error, not vectors!")

    if a.shape[0] < a.shape[1]:
        a = a.transpose()

    if b.shape[0] < b.shape[1]:
        b = b.transpose()

    result = sym.matrices.dense.zeros(rows=a.shape[0], cols=b.shape[0])

    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            result[i, j] = a[i]*b[j]
    
    return result

def vector_sqr(a: Matrix):
    result = 0

    if not 1 in a.shape:
        Print("error, not vectors!")

    if a.shape[0] < a.shape[1]:
        a = a.transpose()

    for i in range(a.shape[0]):
        result += a[i]*a[i]
    
    return result

delta = sym.matrices.dense.eye(rows=3, cols=3)
    
def main():
    Re = ConstantSymbol('Re', value=100)
    Rem = ConstantSymbol('Rem', value=100)
    Ma = ConstantSymbol('Ma', value=1.1)
    gamma = ConstantSymbol('gamma', value=1.4)

    rho = Symbol('rho')
    # velocity
    u = Symbol('u')
    v = Symbol('v')
    w = Symbol('w')
    # magnetic field
    Bx = Symbol('Bx')
    By = Symbol('By')
    Bz = Symbol('Bz')

    U = Matrix([u, v, w])
    B = Matrix([Bx, By, Bz])

    A = Matrix([rho, rho*u, rho*v, rho*w, Bx, By, Bz])

    p = rho ** (gamma)

    Fluxes = []

    for j in range(3):
        flux_j = sym.matrices.dense.zeros(rows=1, cols=7)
        flux_j[0] = rho*U[j]
        for i in range(3):
            flux_j[i + 1] = rho*U[i]*U[j] + ( p + vector_sqr(B) / (2 * Ma**2) ) * delta[i, j] - B[i]*B[j] / Ma**2
        
        for i in range(3):
            flux_j[i + 4] = U[j]*B[i] - U[i]*B[j]

        Fluxes.append(flux_j)

        if j == 0:
            print(flux_j)
            jac = flux_j.jacobian(A)
            print(jac)
            print(jac.eigenvalues())
    
    
if __name__ == "__main__":
    main()