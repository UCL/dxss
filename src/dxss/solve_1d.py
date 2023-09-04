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
import pypardiso
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

class PySolver:
    def __init__(self,Asp,psolver):
        self.Asp = Asp
        self.solver = psolver
    def solve(self,b_inp,x_out): 
        self.solver._check_A(self.Asp)
        b = self.solver._check_b(self.Asp, b_inp.array)
        self.solver.set_phase(33)
        x_out.array[:] = self.solver._call_pardiso(self.Asp , b )[:]

t0 = 0
#T = 1.0
T = 1
N = 32
N_x = int(5*N)
#N_x = int(2*int(N/2))
#N_x = int(2*N)
order = 1
k = order
q = order
#kstar = order
#qstar = order
kstar = 1
if order == 1:
    qstar = 1
else:
    qstar = 0

stabs = {"data": 1e4, 
        "dual": 1.0,
        "primal": 1e-3,
        "primal-jump":1.0,
       } 

#stabs = {"data": 1e0, 
#        "dual": 1.0,
#        "primal": 1e-3,
#        "primal-jump":1.0,
#       } 

# define quantities depending on space
msh = create_unit_interval(MPI.COMM_WORLD, N_x)

def omega_Ind_convex(x):    
    values = np.zeros(x.shape[1],dtype=PETSc.ScalarType)
    omega_coords = np.logical_or( ( x[0] <= 0.2 ), (x[0] >= 0.8 ))  
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    #print("values = ", values)
    return values

def omega_Ind_noGCC(x):    
    values = np.zeros(x.shape[1],dtype=PETSc.ScalarType)
    omega_coords =  ( x[0] <= 0.2 )  
    #omega_coords =  ( x[0] <= 0.45 )  
    #omega_coords =  ( x[0] <= 0.25 )  
    rest_coords = np.invert(omega_coords)
    values[omega_coords] = np.full(sum(omega_coords), 1.0)
    values[rest_coords] = np.full(sum(rest_coords), 0)
    #print("values = ", values)
    return values


def sample_sol(t,xu):
    return ufl.cos(2*pi*(t))*ufl.sin(2*pi*xu[0])

def dt_sample_sol(t,xu):
    return -2*pi*ufl.sin(2*pi*(t))*ufl.sin(2*pi*xu[0])

#def sample_sol(t,xu):
#    return ufl.sin(pi*(t))*ufl.cos( 0.5*pi + xu[0]*pi )

#def dt_sample_sol(t,xu):
#    return pi*ufl.cos(pi*(t))*ufl.cos( 0.5*pi + xu[0]*pi )


#print(sample_sol(0.5,[0.25]))
#input("")

st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=T,t=t0,msh=msh,Omega_Ind=omega_Ind_noGCC,stabs=stabs,sol=sample_sol,dt_sol=dt_sample_sol )
#st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=T,t=t0,msh=msh,Omega_Ind=omega_Ind_convex,stabs=stabs,sol=sample_sol,dt_sol=dt_sample_sol )
st.SetupSpaceTimeFEs()
st.PreparePrecondGMRes()
A_space_time_linop = st.GetSpaceTimeMatrixAsLinearOperator()
b_rhs = st.GetSpaceTimeRhs()
# Prepare the solvers for problems on the slabs 
#st.SetSolverSlab(GetLuSolver(st.msh,st.GetSlabMat())) # general slab
#st.SetSolverFirstSlab(GetLuSolver(st.msh,st.GetSlabMatFirstSlab())) # first slab is special
# GMRes iteration
#u_GMRES,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_time_marching,maxsteps = 100000, tol = 1e-7, startiteration = 0, printrates = True)
#st.MeasureErrors(u_GMRES)


# Prepare coarse grid correction 
#N_coarse = int(N_x/3)
#msh_coarse = create_unit_interval(MPI.COMM_WORLD, N_coarse)
#st.PrepareCoarseGridCorrection(msh_coarse)
#st.SetSolverCoarse(GetLuSolver(msh_coarse,st.GetSpaceTimeMatCoarse())) 
#u_GMRES,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_twolvl,x=None,maxsteps = 100000, tol = 1e-7, innerproduct = None, callback = None, restart = None, startiteration = 0, printrates = True, reltol = None)
#st.MeasureErrors(u_GMRES)

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
        
        #x_sweep_once  = fem.petsc.create_vector(st.SpaceTimeLfi)
        #residual  = fem.petsc.create_vector(st.SpaceTimeLfi)
        #diff = fem.petsc.create_vector(st.SpaceTimeLfi)
        #st.pre_time_marching_improved(b_rhs, x_sweep_once)
        #st.Plot( x_sweep_once ,N_space=500,N_time_subdiv=20)
        #u_sol,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_time_marching_improved,maxsteps = 100000, tol = 1e-7, startiteration = 0, printrates = True)
        u_sol,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_time_marching_improved,maxsteps = 100000, tol = 1e-7, startiteration = 0, printrates = True)
        
        #diff.array[:] = np.abs( x_sweep_once.array[:] -  u_sol.array[:]  ) 
        
        #st.ApplySpaceTimeMatrix(x_sweep_once,residual)
        #residual.array[:] -= b_rhs.array[:] 
        #residual.array[:] = np.abs( residual.array[:] )
        #print("plotting residual")
        #st.Plot(residual,N_space=500,N_time_subdiv=20,abs_val=True)
        #print("plotting diff")
        #u_sol,res = GMRes(A_space_time_linop,b_rhs,pre=st.pre_Schwarz,maxsteps = 100000, tol = 1e-7, startiteration = 0, printrates = True)
        #st.Plot(diff,N_space=500,N_time_subdiv=20,abs_val=True)
        #st.Plot(u_sol,N_space=500,N_time_subdiv=20)

        st.PlotError(u_sol,N_space=500,N_time_subdiv=20)
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

    #st.Plot(N_space=500,N_time_subdiv=20)
