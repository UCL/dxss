import numpy as np
from math import pi,sqrt  
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, dS,jump,div
from mpi4py import MPI


from petsc4py import PETSc

from dolfinx.mesh import create_unit_interval
from dolfinx.io import XDMFFile
import sys
sys.setrecursionlimit(10**6)
from dxss.gmres import GMRes
from dxss.space_time import * 
from dxss.precomp_time_int import theta_ref, d_theta_ref 
from dxss.meshes import get_mesh_hierarchy, get_mesh_data_all_around
import pypardiso
import scipy.sparse as sp
import time
import cProfile
import resource



#solver_type = "petsc-LU"  
solver_type = "pypardiso" # 
#solver_type = "direct" # 

def GetLuSolver(msh,mat):
    solver = PETSc.KSP().create(msh.comm) 
    solver.setOperators(mat)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    return solver

# define alternative solvers here 
#def GetSpMat(mat):
#    ai, aj, av = mat.getValuesCSR()
#    Asp = sp.csr_matrix((av, aj, ai))
#    return Asp 


#class PySolver:
#    def __init__(self,Asp,psolver):
#        self.Asp = Asp
#        self.solver = psolver
#    def solve(self,b_inp,x_out):
#        x_py = self.solver.solve(self.Asp, b_inp.array )
#        x_out.array[:] = x_py[:]

class PySolver:
    def __init__(self,Asp,psolver):
        self.Asp = Asp
        self.solver = psolver
    def solve(self,b_inp,x_out): 
        self.solver._check_A(self.Asp)
        b = self.solver._check_b(self.Asp, b_inp.array)
        self.solver.set_phase(33)
        x_out.array[:] = self.solver._call_pardiso(self.Asp , b )[:]



ref_lvl_to_N = [1,2,4,8,16,32]
ref_lvl = 3

t0 = 0
T = 1.0
N = ref_lvl_to_N[ref_lvl]
order = 3
k = order
q = order
kstar = order
qstar = order 
stabs = {"data": 1e4, 
        "dual": 1.0,
        "primal": 1e-3,
        "primal-jump":1.0,
       } 

# define quantities depending on space
ls_mesh = get_mesh_data_all_around(5,init_h_scale=5.0)
#for j in range(len(ls_mesh)):
#    with io.XDMFFile(ls_mesh[j].comm, "mesh-reflvl{0}.xdmf".format(j), "w") as xdmf:
#        xdmf.write_mesh(ls_mesh[j])
msh = ls_mesh[ref_lvl]

def omega_Ind_convex(x): 
    values = np.zeros(x.shape[1],dtype=PETSc.ScalarType)
    omega_coords = np.logical_or( ( x[0] <= 0.2 ), 
      np.logical_or(  ( x[0] >= 0.8 ),        
        np.logical_or(   (x[1] >= 0.8 ), (x[1] <= 0.2)  )
        )
      )
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    return values

def sample_sol(t,xu):
    return ufl.cos(sqrt(2)*pi*t)*ufl.sin(pi*xu[0])*ufl.sin(pi*xu[1])

def dt_sample_sol(t,xu):
    return -sqrt(2)*pi*ufl.sin(sqrt(2)*pi*t)*ufl.sin(pi*xu[0])*ufl.sin(pi*xu[1])

st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=T,t=t0,msh=msh,Omega_Ind=omega_Ind_convex,stabs=stabs,sol=sample_sol,dt_sol=dt_sample_sol)
st.SetupSpaceTimeFEs()
st.PreparePrecondGMRes()
A_space_time_linop = st.GetSpaceTimeMatrixAsLinearOperator()
b_rhs = st.GetSpaceTimeRhs()

# Prepare the solvers for problems on the slabs 
#st.SetSolverSlab(GetLuSolver(st.msh,st.GetSlabMat())) # general slab
#st.SetSolverFirstSlab(GetLuSolver(st.msh,st.GetSlabMatFirstSlab())) # first slab is special


def SolveProblem(measure_errors = False):

    start=time.time()
    if solver_type == "pypardiso":
         
        genreal_slab_solver = pypardiso.PyPardisoSolver()
        SlabMatSp = GetSpMat(st.GetSlabMat())
        
        genreal_slab_solver.factorize( SlabMatSp)   
        st.SetSolverSlab(PySolver(SlabMatSp, genreal_slab_solver))
        
        initial_slab_solver = pypardiso.PyPardisoSolver()
        SlabMatFirstSlabSp = GetSpMat( st.GetSlabMatFirstSlab())   
        initial_slab_solver.factorize( SlabMatFirstSlabSp  )   
        st.SetSolverFirstSlab(PySolver( SlabMatFirstSlabSp,  initial_slab_solver ))
         
        #st.SetSolverSlab( PySolver ( GetSpMat( st.GetSlabMat()))  )   # general slab
        #st.SetSolverFirstSlab( PySolver ( GetSpMat( st.GetSlabMatFirstSlab()) ) ) 
        #u_sol,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_time_marching,maxsteps = 100000, tol = 1e-7, startiteration = 0, printrates = True)
        u_sol,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_time_marching_improved,maxsteps = 100000, tol = 1e-7, startiteration = 0, printrates = True)
    elif solver_type == "petsc-LU":
        st.SetSolverSlab(GetLuSolver(st.msh,st.GetSlabMat())) # general slab
        st.SetSolverFirstSlab(GetLuSolver(st.msh,st.GetSlabMatFirstSlab())) # first slab is special
        u_sol,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_time_marching_improved,maxsteps = 100000, tol = 1e-7, startiteration = 0, printrates = True)
    else: 
        A_space_time = st.GetSpaceTimeMatrix() 
        u_sol,_ = A_space_time.createVecs() 
        solver_space_time = GetLuSolver(st.msh,A_space_time)
        solver_space_time.solve(u_sol ,b_rhs)
   
    end=time.time()
    print("elapsed time  " + str(end-start)+ " seconds")

    if measure_errors:
        st.MeasureErrors(u_sol)

if __name__ == "__main__":

    #cProfile.run('SolveProblem()')
    SolveProblem(measure_errors = True) 

    print("Memory usage in (Gb) = ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6 )


