
import numpy as np



def zero_init(shape):
    return np.zeros(shape)

def random_init( shape):
    return np.random.randn(*shape)

def xavier_init(shape):
    in_dim, out_dim = shape
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim)) 
    return np.random.randn(in_dim, out_dim) * xavier_stddev

def he_init(shape):
    in_dim = shape[1]
    out_dim = shape[0]
    he_stddev = np.sqrt(2 / in_dim)
    return np.random.randn(in_dim, out_dim) * he_stddev

def normal_init(shape, mean=0.0, stddev=0.1):
    return np.random.normal(mean, stddev, shape)

def uniform_init(shape, a=-0.1, b=0.1):
    return np.random.uniform(a, b, shape)

def lecun_init(shape):
    in_dim = shape[1]
    stddev = 1 / np.sqrt(in_dim)
    return np.random.normal(0.0, stddev, shape)

def orthogonal_init(shape, gain=1.0):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return gain * q

def kaiming_init(shape):
    in_dim = shape[1]
    stddev = np.sqrt(2.0 / in_dim)
    return np.random.normal(0.0, stddev, shape)


if __name__ == '__main__':
    print("zero_init")
    img = zero_init((3, 3))
    print(img)
    print("random_init")
    img = random_init((3, 3))
    print(img)
    print("xavier_init")
    img = xavier_init((3, 3))
    print(img)
    print("he_init")
    img = he_init((3, 3))
    print(img)
    print("normal_init")
    img = normal_init((3, 3))
    print(img)
    print("uniform_init")
    img = uniform_init((3, 3))
    print(img)
    print("lecun_init")
    img = lecun_init((3, 3))
    print(img)
    print("orthogonal_init")
    img = orthogonal_init((3, 3))
    print(img)
    print("kaiming_init")
    img = kaiming_init((3, 3))
    print(img)
    