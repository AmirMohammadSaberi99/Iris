$ python
>>> import numpy as np
>>> data = np.load('iris_feats.npz')
>>> print(data.files)
['gabor', 'wavelet']
>>> gabor = data['gabor']
>>> wavelet = data['wavelet']
>>> print('Gabor shape:', gabor.shape)
>>> print('Wavelet shape:', wavelet.shape)
>>> # Peek at the first few values
>>> print(gabor[:10])
