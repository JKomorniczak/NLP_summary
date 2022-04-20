import numpy as np

file = np.load('vect/tfidf/business/004.npy')
print(file)

np.savetxt("foo.csv", file, delimiter=",", fmt="%.3f")