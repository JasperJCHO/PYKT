import numpy as np

a = np.zeros((10, 2))
print(a)
b = a.T
print('b==',b)
c = b.view()
print('c==',c)
print(a.shape, b.shape, c.shape)
d = np.reshape(b, (5, 4))
print('d==',d)
e = np.reshape(b, (20,))
print('e==',e)
f = np.reshape(b, (20, -1))
print('f==',f)
g = np.reshape(b, (-1, 20))
print('g==',g)
print(d.shape, e.shape, f.shape, g.shape)