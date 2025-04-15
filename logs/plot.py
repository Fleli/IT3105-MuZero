import numpy as np
import matplotlib.pyplot as plt

f = open('logs/2025-04-15 17-04-46.log', mode='r')

lines = f.readlines()

values = [ int(l.split(':')[1][:-1]) for l in lines ]

k = 250
smooth = []
for i in range(len(values) - k):
    smooth.append( sum(values[i:i+k]) / k )

# plt.plot(values)
plt.plot(smooth)
plt.show()

f.close()

