import numpy as np
from math import pi,sqrt  
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner, dS,jump,div
from mpi4py import MPI


from petsc4py import PETSc

from dolfinx.mesh import create_unit_interval,create_box,CellType, GhostMode
from dolfinx.io import XDMFFile
import sys
sys.setrecursionlimit(10**6)
from dxss.gmres import GMRes
from dxss.space_time import * 
from dxss.precomp_time_int import theta_ref, d_theta_ref 
from dxss.meshes import get_mesh_hierarchy, get_mesh_data_all_around, get_3Dmesh_data_all_around
try:
    import pypardiso
    solver_type = "pypardiso" # 
except ImportError:
    solver_type = "petsc-LU"  

import scipy.sparse as sp
import time
import cProfile
import resource



#solver_type = "direct" # 

GCC = False 


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



ref_lvl_to_N = [1,2,4,8,16,32]
ref_lvl = 3

data_size = 0.25

t0 = 0
T = 1/2
N = ref_lvl_to_N[ref_lvl]
order = 1
k = order
q = order
kstar = order
qstar = order 
stabs = {"data": 1e4, 
        "dual": 1.0,
        "primal": 1e-3,
        "primal-jump":1.0,
       } 

#Nxs = [5,10,20,40]
Nxs = [2,4,8,16,32,64]

Nx = Nxs[ref_lvl]
print("Nx = ", Nx)
# define quantities depending on space
#ls_mesh = get_3Dmesh_data_all_around(ref_lvl,init_h_scale=5.0)
#for j in range(len(ls_mesh)):
#    with io.XDMFFile(ls_mesh[j].comm, "mesh-reflvl{0}.xdmf".format(j), "w") as xdmf:
#        xdmf.write_mesh(ls_mesh[j])
#input("")
#msh = ls_mesh[ref_lvl]

msh = create_box(MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
                                  np.array([1.0, 1.0, 1.0])], [Nx, Nx, Nx],
                 CellType.hexahedron, ghost_mode=GhostMode.shared_facet)


#with io.XDMFFile(msh.comm, "msh-hex-reflvl{0}.xdmf".format(ref_lvl), "w") as xdmf:
#    xdmf.write_mesh( msh )
#input("")
if GCC:
    if Nx > 2:
        def omega_Ind(x): 
            values = np.zeros(x.shape[1],dtype=PETSc.ScalarType)
            omega_coords = np.logical_or( ( x[0] <= data_size  ), 
                           np.logical_or(  ( x[0] >= 1.0-data_size  ),        
                           np.logical_or(   (x[1] >= 1.0-data_size  ), 
                           np.logical_or(    (x[1] <= data_size  ),            
                           np.logical_or(    (x[2] <= data_size  ),(x[2] >= 1.0-data_size  ) )
                              )
                             )
                            )
                           )
            rest_coords = np.invert(omega_coords)
            values[omega_coords] = np.full(sum(omega_coords), 1.0)
            values[rest_coords] = np.full(sum(rest_coords), 0)
            return values
    else:
        x = ufl.SpatialCoordinate(msh)
        omega_indicator = ufl.Not(
                           ufl.And(
                            ufl.And(x[0] >= data_size, x[0] <= 1.0-data_size), 
                                   ufl.And(ufl.And(x[1] >= data_size, x[1] <= 1.0-data_size),
                                           ufl.And( x[2] >= data_size, x[2] <= 1.0-data_size)
                                          )
                                  )
                                 )
        omega_Ind = ufl.conditional(omega_indicator, 1, 0)
else:
    if Nx > 2:
        def omega_Ind(x): 
            values = np.zeros(x.shape[1],dtype=PETSc.ScalarType)
            omega_coords = np.logical_and( ( x[0] <= data_size  ), ( x[0] >= 0.0 ))  
            rest_coords = np.invert(omega_coords)
            values[omega_coords] = np.full(sum(omega_coords), 1.0)
            values[rest_coords] = np.full(sum(rest_coords), 0)
            return values
    else:
        x = ufl.SpatialCoordinate(msh)
        omega_indicator = ufl.And(x[0] <= data_size, x[0] >= 0.0) 
        omega_Ind = ufl.conditional(omega_indicator, 1, 0)


def sample_sol(t,xu):
    return ufl.cos(sqrt(3)*pi*t)*ufl.sin(pi*xu[0])*ufl.sin(pi*xu[1])*ufl.sin(pi*xu[2])

def dt_sample_sol(t,xu):
    return -sqrt(3)*pi*ufl.sin(sqrt(3)*pi*t)*ufl.sin(pi*xu[0])*ufl.sin(pi*xu[1])*ufl.sin(pi*xu[2])


st = space_time(q=q,qstar=qstar,k=k,kstar=kstar,N=N,T=T,t=t0,msh=msh,Omega_Ind=omega_Ind,stabs=stabs,sol=sample_sol,dt_sol=dt_sample_sol,data_dom_fitted= Nx > 2 )
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
        #st.SolverFirstSlab = st.SolverSlab
         
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

        st.PlotParaview(u_sol,name="abserr-cube-GCC-order{0}".format(order)) 
        st.MeasureErrors(u_sol)

if __name__ == "__main__":

    #cProfile.run('SolveProblem()')
    SolveProblem(measure_errors = True) 


    print("Memory usage in (Gb) = ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6 )


