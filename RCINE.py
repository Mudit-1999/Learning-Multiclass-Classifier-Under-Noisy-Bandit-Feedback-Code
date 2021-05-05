from RCNBF import *
from NREst import *
import numpy as np


# Steps to Run on a new dataset
# 1. Add the datafiles to the dataset folder and write a parser
#    which read all the datafile
# 2. Set the input parameter acc to the dataset 
# 3. If you want to add noise in the dataset, then set it to the 
#    specific value otherwise set it to zero
# 4. Now go to the NREst file for further instuction
 

# Fashion Mnist
# file1="./dataset/mnist.csv"

# file2="./dataset/mnist.csv"
# SynSep
# file6="./dataset/synsep.csv"
# IRIS
# file3="./dataset/iris.data"
# usps
file4="./dataset/upsp.csv"





dataset=[]
# IRIS Dataset
# with open(file3,'r') as f:
#     lines=f.readlines()
#     for i in lines:
#         temp=i.strip().split(',')
#         tmp1=[]
#         # print(" dwd ",len(temp))
#         if len(temp)<4:
#             continue
#         for field in temp:
#             if field=="Iris-setosa":
#                 tmp1.append(0)
#             elif field=="Iris-versicolor":
#                 tmp1.append(1)
#             elif field=="Iris-virginica":
#                 tmp1.append(2)                    
#             else:   
#                 tmp1.append(float(field))
#         dataset.append(tmp1)

# MNIST Data
# with open(file2,'r') as f:
#     lines=f.readlines()
#     for i in lines:
#         temp=i.strip().split(',')
#         dataset.append([float(i) for i in temp])
# with open(file5,'r') as f:
#     lines=f.readlines()
#     for i in lines:
#         temp=i.strip().split(',')
#         dataset.append([float(i) for i in temp])

# SynSep Data
# with open(file6,'r') as f:
#     lines=f.readlines()
#     for i in lines:
#         temp=i.strip().split(',')
#         dataset.append([float(i) for i in temp])


# USPS Data
with open(file4,'r') as f:
    lines=f.readlines()
    for i in lines:
        temp=i.strip().split(',')
        dataset.append([float(i) for i in temp])

# Input parament
dataset_name="usps"
size=int(50000*200) # number of iterations 
k=10                # class size 
d=256               # instance vector size
g=0.03              # exploration parameter 


# Noise Rates 
real_rho0,real_rho1 = 0.15,0.15 # real noise rates 



# Initialization 
weight_matrix,incorrect_classified,correct_classified,error_rate_list=np.zeros((k,d)),0,0,np.zeros(size)
noise_mat=np.zeros(2)
rho0=0.0
rho1=0.0         

offset=0
# print(len(dataset),len(dataset[0]))
# RCINE algorithm 
for rep in range(200):
    weight_matrix,incorrect_classified,correct_classified,error_rate_list,noisy_data=Run(offset,k,d,g,dataset,size,real_rho0,real_rho1,rho0,rho1,weight_matrix,incorrect_classified,correct_classified,error_rate_list)
    noise_mat=train_and_evaluate(dataset,'crossentropy',data=noisy_data)  
    rho0=noise_mat[0,1]
    rho1=noise_mat[1,0]
    offset+=50000


# Plotting Error Rate Vs Number of Examples 

fig,ax=plt.subplots()
ax.grid()
ax.set_yscale('log')
ax.set_xscale('log')
indx=np.arange(0,size,1000,int)
print(indx[-1])
ax.plot(indx, np.array(error_rate_list)[indx],linestyle='dashed',color='b' ,label='RCINE',marker='*',markevery=1/10)
plt.title(str(dataset_name) + "(\u03B3 = " +str(g_val[0])+")", fontsize=13)
plt.xlabel("Number of Examples",fontsize=13)
plt.ylabel("Error Rate",fontsize=13)
ax.legend(prop={'size' : 13})
plt.show()