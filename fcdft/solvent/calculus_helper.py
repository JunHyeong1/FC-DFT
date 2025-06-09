import numpy as np
from pyamg.gallery.stencil import stencil_grid

def poisson(grid, dtype=float, format=None, type='FD'):
    """Positive definite Laplacian operator generator

    Args:
        grid (_type_): _description_
        dtype (_type_, optional): _description_. Defaults to float.
        format (_type_, optional): _description_. Defaults to None.
        type (str, optional): _description_. Defaults to 'FD'.

    Raises:
        ValueError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    grid = tuple(grid)

    N = len(grid)  # grid dimension

    if N < 1 or min(grid) < 1:
        raise ValueError(f'Invalid grid shape: {grid}')

    # create N-dimension Laplacian stencil
    if type == 'FD':
        # Eighth-order finite difference method
        stencil = np.zeros((9,) * N, dtype=dtype)
        for i in range(N):
            stencil[(4,)*i + (0,) + (4,)*(N-i-1)] = 1/560
            stencil[(4,)*i + (1,) + (4,)*(N-i-1)] = -8/315
            stencil[(4,)*i + (2,) + (4,)*(N-i-1)] = 1/5
            stencil[(4,)*i + (3,) + (4,)*(N-i-1)] = -8/5
            stencil[(4,)*i + (5,) + (4,)*(N-i-1)] = -8/5
            stencil[(4,)*i + (6,) + (4,)*(N-i-1)] = 1/5
            stencil[(4,)*i + (7,) + (4,)*(N-i-1)] = -8/315
            stencil[(4,)*i + (8,) + (4,)*(N-i-1)] = 1/560
        stencil[(4,)*N] = 205/72*N

    if type == 'FE':
        raise NotImplementedError

    return stencil_grid(stencil, grid, format=format)

def gradient(f, *varargs):
    """Optimized Gradient Calculator"""
    N = len(f.shape)  # number of dimensions
    n = len(varargs)
    if n == 0:
        dx = [1.0]*N
    elif n == 1:
        dx = [varargs[0]]*N
    elif n == N:
        dx = list(varargs)
    else:
        raise SyntaxError("invalid number of arguments")

    outvals = []
    weights = np.array([3, -32, 168, -672, 672, -168, 32, -3])
    forward = np.array([-3, 16, -36, 48, -25])
    for axis in range(N):
        out = np.zeros_like(f)
        slices = [slice(None)] * N
        slices[axis] = slice(4, -4)
        out[tuple(slices)] = (weights[0] * f[(slice(None),) * axis + (slice(None, -8),)] +
                    weights[1] * f[(slice(None),) * axis + (slice(1, -7),)] +
                    weights[2] * f[(slice(None),) * axis + (slice(2, -6),)] +
                    weights[3] * f[(slice(None),) * axis + (slice(3, -5),)] +
                    weights[4] * f[(slice(None),) * axis + (slice(5, -3),)] +
                    weights[5] * f[(slice(None),) * axis + (slice(6, -2),)] +
                    weights[6] * f[(slice(None),) * axis + (slice(7, -1),)] +
                    weights[7] * f[(slice(None),) * axis + (slice(8, None),)]
                    ) / 840.0
        
        slices[axis] = slice(None, 4)
        out[tuple(slices)] = (forward[0]*f.take(range(4, 8), axis) + 
                              forward[1]*f.take(range(3, 7), axis) + 
                              forward[2]*f.take(range(2, 6), axis) + 
                              forward[3]*f.take(range(1, 5), axis) + 
                              forward[4]*f.take(range(0, 4), axis)) / 12.0
        
        slices[axis] = slice(-4, None)
        out[tuple(slices)] = -(forward[4]*f.take(range(-4,  0), axis) +
                               forward[3]*f.take(range(-5, -1), axis) + 
                               forward[2]*f.take(range(-6, -2), axis) + 
                               forward[1]*f.take(range(-7, -3), axis) + 
                               forward[0]*f.take(range(-8, -4), axis)) / 12.0
        outvals.append(out / dx[axis])

    if N == 1:
        return outvals[0]
    else:
        return outvals

def vectorize_grad(grad_object):
    grad = np.column_stack((grad_object[0].ravel(), grad_object[1].ravel(), grad_object[2].ravel()))
    return grad