import numpy as np

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def process_label(label, cmap):
    
    result = np.zeros((label.shape[0], label.shape[1]), dtype = np.uint8)
    
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            ok = False
            
            for index in range(len(cmap)):
                if np.array_equal(label[i,j],cmap[index]):
                    result[i,j] = index
                    ok = True
                    
            if not ok:
                result[i,j] = 21
            
    return result