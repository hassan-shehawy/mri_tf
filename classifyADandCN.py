import scipy.io as scird
import tensorflow as tf

cnmat = scird.loadmat('CNcounts141.mat')
CNcounts=cnmat['CNcounts']

admat = scird.loadmat('ADcounts141.mat')
ADcounts=admat['ADcounts']

print(ADcounts[0:4,4,6])


