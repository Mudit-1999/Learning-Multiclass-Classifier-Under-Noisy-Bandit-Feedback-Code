@jit(nopython=True)
def Noisy_feedback(true_label,predicted_label,real_rho0,real_rho1):
    fb=0
    cutoff=random.random()
    if true_label==predicted_label:
        fb=1
    if fb==1:
        if cutoff<real_rho1:
            fb=0
    else:
        if cutoff<real_rho0:
            fb=1
    return fb


@jit(nopython=True)
def cal_prob(cal_label,gamma):
    prob= np.ones(k)
    prob = prob*gamma/k
    prob[cal_label]+= 1 -gamma
    # print("Prob",prob)
    return prob

@jit(nopython=True)
def random_sample(prob):
    number = float32(random.random()) * np.sum(prob)
    # print("Sum prob",sum(prob), number)
    for i in range(0,prob.shape[0]):
        if number < prob[i]:
            return i
        number -= prob[i]
    return prob.shape[0]-1




@jit(nopython=True)
def Run(offset,k,d,gamma,size,real_rho0=0,real_rho1=0,rho0=0,rho1=0,weight_matrix=None,incorrect_classified=0,correct_classified=0,error_rate_list=None): # feature_vector 1xd
    weight_matrix=weight_matrix
    incorrect_classified=incorrect_classified
    error_rate=0
    correct_classified=correct_classified
    error_rate_list=error_rate_list
    beta= 1- rho0 -rho1
    noisy_data=np.zeros((50000,d+2))
    tnoise=1
    tind=0
    for i in range(0,50000):
        num=random.randint(0,data.shape[0]-1)
        entry= data[num,:]
        feature_vector=np.reshape(entry[0:d],(d,1))
        # print(i)
        true_label=int(entry[d])
        y_tilde=int(-1)
    
        cal_label=np.argmax(np.reshape(np.dot(weight_matrix,feature_vector),-1))   # calculated label
        prob=cal_prob(cal_label,gamma)     
        y_tilde= random_sample(prob) #predicted label 
        # print(cal_label)
        cfb=Noisy_feedback(true_label,y_tilde,real_rho0,real_rho1)

        basis_vec=np.zeros((k,1))
        basis_vec[y_tilde,0]=-rho0/(beta*prob[y_tilde])
        if cfb==1:
            basis_vec[y_tilde,0]+=1/(beta*prob[y_tilde])
        basis_vec[cal_label,0]-=1

        weight_matrix+=np.kron(basis_vec,feature_vector.T)
        noisy_data[tind,0:d]=feature_vector.T
        noisy_data[tind,d]=y_tilde
        noisy_data[tind,d+1]=cfb
        tind+=1
        if true_label==y_tilde:    
            correct_classified+=1
        else:
            incorrect_classified+=1
        #Cal error rate 
        error_rate=incorrect_classified/(incorrect_classified+correct_classified)
        error_rate_list[i+offset] = error_rate
        # print(error_rate)
    print("At",i+offset , "(RB) g:",gamma, "real_rho", real_rho0, "rho" , rho0 ,"real_rh1", real_rho1, "rh1" , rho1 ,"error_rate",error_rate)
    return weight_matrix,incorrect_classified,correct_classified,error_rate_list,noisy_data