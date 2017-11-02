

from mpi4py import MPI
import mpi
import math as m

comm = MPI.COMM_WORLD
ncpu = comm.size
rank = comm.Get_rank()

def action(x):
	sqr = lambda y : m.sin(m.cos(m.sin(m.cos(y*y))))*m.sin(m.cos(m.sin(m.cos(y*y))))
	return sqr(x)

if rank == 0:
	task = [[i+j*10+1 for i in range(1000)] for j in range(100)]
	num = 2


else:
	task = None
	num = None


num = comm.bcast(num, root=0)
results = mpi.control(action, task)
if rank == 0:
	print(results)
else:
	print(rank)

