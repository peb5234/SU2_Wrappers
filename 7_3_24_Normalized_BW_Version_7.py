#Standardized and Normalized Wrapper Test
#7/3/24

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import pandas as pd
import numpy as np
import subprocess
import argparse
import warnings
import time
from botorch.utils.transforms import standardize, normalize, unnormalize

warnings.filterwarnings('ignore')

#Wrapper Class
class ExperimentDesignWrapper():
    #Initializer
    def __init__(self):
        self.input_file=None
        self.config_file=None
        return
    
    #Method to read data from a CSV file and return a numpy array
    def read_input_csv_file(self,input_csv_file):
        input_csv_dataframe=pd.read_csv(input_csv_file,header=None)
        input_numpy_array=input_csv_dataframe.to_numpy()
        return input_numpy_array
    
    #Method to write the input file data into the Config Files
    def write_config_file(self,config_file,inputs):
        
        headers=inputs[0]
        params=inputs[1]

        #Opening and storing the config file data passed
        with open(config_file,'r') as infile:
            config_file_data=infile.readlines()

        #opening and rewriting the config file
        with open(config_file,'w') as outfile: 
            for line in config_file_data:
            
                match=False

                #Overwriting desired lines
                for index, header in enumerate(headers): #Checking the header
                    if line.startswith(header):
                        Label,old_value=line.split('=')
                        newline=f"{Label}={params[index]}\n"
                        outfile.write(newline)
                        match=True

                #Other lines 
                if not match:
                    outfile.write(line)
            

        return 
    
    #Method to write the design values into the cfg files
    def write_config_file_DVs(self,config_file,dv_values):

        #Opening and storing the config file data passed
        with open(config_file,'r') as infile:
            config_file_data=infile.readlines()

        #opening and rewriting the config file
        with open(config_file,'w') as outfile: 
            for line in config_file_data:
                if line.startswith("DV_VALUE"):
                    Label, old_value=line.split('=')
                    joined_dv=','.join(str(coefficient) for coefficient in dv_values)
                    newline=f"{Label}={joined_dv}\n"
                    outfile.write(newline)
                    #if newline.startswith("DV_VALUE="):
                        #print(newline)
                else:
                    outfile.write(line)
        
        return
    
 

    #Method to run SU2 for mesh deformation
    def run_SU2_DEF(self,config_file,num_processes=1):
        subprocess.run(['/storage/icds/RISE/sw8/su2-7.5.1/bin/SU2_DEF',config_file])

    #Method to run SU2 CFD simulation
    def run_SU2_CFD(self,config_file,num_processes=1):
        subprocess.run(['/storage/icds/RISE/sw8/su2-7.5.1/bin/SU2_CFD',config_file])

    #Method to get the CD CL and CF from the history CSV file
    def get_output_data(self,history_output):
        history_output_dataframe=pd.read_csv(history_output) 
        cd = history_output_dataframe['       "CD"       '].iloc[-1]
        #print(f"cd from History.csv is: {cd}\n")
        cl = history_output_dataframe['       "CL"       '].iloc[-1]
        cmz = history_output_dataframe['       "CMz"      '].iloc[-1]
        
        return cd,cl,cmz



#Objective Function Block

def objective_function(dv_value,i):
    
    print(f"DV_VALUE (new_x) is {dv_value} when passed to the objective_function \n")
    
    EDW=ExperimentDesignWrapper()

    Cd=[]
    
    EDW.write_config_file_DVs(meshdef_cfg,dv_value)
    print("\n Running SU2_DEF in Obj_func")
    starttime=time.time()
    EDW.run_SU2_DEF(meshdef_cfg,num_processes)
    subprocess.run(["cp", "surface_deformed.vtu", "surface_deformed_" + str(i) + ".vtu"])
    endtime=time.time()
    timediff=endtime-starttime
    print(f"SU2_DEF took {timediff} seconds \n")
    #subprocess.run(["cp", "mesh_out.su2", "mesh_deformed.su2"]) #copy the deformed mesh to mesh_deformed.su2 for running again

    #EDW.write_config_file(primal_cfg,[parameters[0,:], parameters[i+1,:]])
    EDW.write_config_file_DVs(primal_cfg,dv_value)

    print("\n Running SU2_CFD in Obj_func")
    starttime=time.time()
    EDW.run_SU2_CFD(primal_cfg,num_processes)
    endtime=time.time()
    timediff=endtime-starttime
    print(f"SU2_CFD took {timediff} seconds \n")
    
    #Getting the CD,CL,CMZ from each loop and storing it
    temp_cd,temp_cl,temp_cmz=EDW.get_output_data(history_file)
    negative_cd=-temp_cd
    Cd.append(negative_cd)
    #print(f"Cd from obj_function is: {Cd}\n")
    #Cl.append(temp_cl)
    #CMz.append(temp_cmz)

    #File Saving CD,CL,CMZ
    return Cd
        

## Bayesian Section ##

def BayesianOptimizationLoop():
    
    print(f"\n\n############################## INITIALIZATION ##############################\n\n")
    
    # Define the search space for the FFD control points
    bounds = torch.tensor([
        [-.0005, -.0005, -.0005, -.0005],  # lower bounds for [1,2,3,4]
        [.0005,.0005,.0005,.0005]   # upper bounds for [1,2,3,4]
    ], dtype=torch.float64)
    
    gp_bounds= torch.stack([torch.zeros(4), torch.ones(4)])
    
    # Initialize data
    train_x = torch.rand(5, 4) * (bounds[1] - bounds[0]) + bounds[0]  # Initial random samples
    
    #Normalizing to [0,1]
    train_x_normalized=normalize(train_x, bounds=bounds)
    print("Before Initialzation Function \n")
    train_y = torch.tensor([objective_function(x.tolist(),'init') for x in train_x])
    print("After Initialization Function\n")
    
   
    
    train_y_standardized = standardize(train_y)
    
    print(f"Train_X initiated as: \n{train_x}\n")
    print(f"Train_x_normalized initiated as: \n{train_x_normalized}\n")
    print(f"Train_Y initiated as: \n{train_y}\n")
    print(f"Train_y_standardized initiated as: \n{train_y_standardized}\n")
    
    ITERS=10

    # Bayesian Optimization Loop
    for iteration in range(ITERS):  # Adjust number of iterations as needed
        print(f"\n\n############################## ITERATION {iteration+1} ##############################\n\n")
        # Fit the GP model
        gp = SingleTaskGP(train_x_normalized, train_y_standardized)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        # Define the acquisition function
        UCB = UpperConfidenceBound(gp, beta=2)
        
        # Optimize the acquisition function to get new candidate (new_x is normalized here)
        print("Calculating new_x\n")
        new_x, _ = optimize_acqf(
            acq_function=UCB,
            bounds=gp_bounds,
            q=1,
            num_restarts=20,
            raw_samples=1024,
            
        )
        
        #Print Statements
        print(f"new_x (normalized) is: \n{new_x}\n")
        
        # Update training data
        train_x_normalized= torch.cat([train_x_normalized, new_x]) #Normalized Train_x updating
        train_x=unnormalize(train_x_normalized,bounds=bounds) #setting train_x to be unnormalized version to pass into new_y
        
        print(f"new_x (unnormalized) is: \n{train_x[iteration+5]}\n")
        
        # Evaluate the objective function
        print("calculating new_y \n")
        new_y = torch.tensor([[objective_function(train_x[iteration+5].tolist(),iteration)]]).reshape(-1, 1)
        print("new_y calculated \n")
        
        
        
        train_y = torch.cat([train_y, new_y])
        train_y_standardized=standardize(train_y)
       
        #Print training updated data
        print(f"train_x_normalized is:\n{train_x_normalized}\n")
        print(f"train_x after unnormalize is:\n{train_x}\n")
        print(f"train_y is:\n{train_y}\n")
        print(f"train_y_standarized after standardize is:\n{train_y_standardized}\n")

        #Print Statements
        print(f"new_y is: \n{new_y}\n")
              
        subprocess.run(["cp", "ffd_boxes_def_0.vtk", "ffd_boxes_def_" + str(iteration) + ".vtk"])
        #subprocess.run(["cp", "ffd_boxes_0.vtk", "ffd_boxes_" + str(iteration) + ".vtk"])
        
        
        print(f"\nIteration {iteration + 1}: Best drag coefficient so far = {train_y.max()}\n")

    # Best found configuration
    #best_idx = train_y.argmin()
    best_idx=train_y.argmax()
    best_x = train_x[best_idx]
    best_y =train_y[best_idx]
    #index = torch.where(train_y == best_y)
    #best_x=train_x[index]
    

    np.save('Best_Shape',best_x)
    np.save('Best_Drag',best_y)
    print("Best FFD control points:", best_x.tolist())
    print("Minimum drag coefficient:", -best_y)
    print("Best Index: ", best_idx-5)

    
    
    
#Parser Block

#Setting parser arguments
parser = argparse.ArgumentParser()
parser.add_argument("-bump","--bump_csv",type=str, default="DV_VALUES.csv", help='Path to the input csv file containing DV_VALUE information')
parser.add_argument("-params","--parameters_csv",type=str, default="CFG_Parameters.csv", help='Path to the input csv file containing simulation parameters')
parser.add_argument("-pcfg","--primal_cfg",type=str, default='ffd_rae_deformed_forward_COPY.cfg', help='Path to the primal cfg file')
parser.add_argument("-mdcfg","--meshdef_cfg",type=str, default='ffd_rae_COPY_.cfg', help='Path to the mesh deformation cfg file')
parser.add_argument("-si","--start_iters",type=int, default=0, help='Number of iterations to start at')
parser.add_argument("-np","--num_procs",type=int,default=1,help='Number of processes')
parser.add_argument("-hf","--history_file",type=str,default='history.csv',help="History file containing aerodynamic coefficient and convergence history")

args = parser.parse_args()

#Saving Inputs from parser to variables
bump_csv=args.bump_csv
params_csv=args.parameters_csv
primal_cfg=args.primal_cfg
meshdef_cfg=args.meshdef_cfg
start_iters=args.start_iters
num_processes=args.num_procs
history_file=args.history_file


#Copying the Files used to ensure they are correctly written at the start:

subprocess.run(["cp", "ffd_rae_deformed_forward.cfg", "ffd_rae_deformed_forward_COPY.cfg"])
subprocess.run(["cp","ffd_rae_.cfg","ffd_rae_COPY_.cfg"])

#Control Block

subprocess.run(["cp", "~/work/SU2/SU2-utils/rae2822/rae_freestream_ffd/mesh_deformed.su2", "mesh_deformed.su2"])
BayesianOptimizationLoop() #Calling the function
print('Finito')