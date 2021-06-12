




#%% Loading packages and data

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import math

#%% Simulation data (one patient)

# Setting - time series:  
#     Outcome:             Y (one-demensional outcome first considered)
#     Treatments:          A_vec = [a1, ..., ak] 
#     Observed covariates: X_vec = [x1, ..., xp]
#     Hidden confounders:  L_vec = [l1, ..., lr]

# Simulation setup:
#     l_(t,j) = f_l( l_(t-i,j), a_(t-i,u) )       for i = 1,...,h (lag h), j = 1,...,r and u = 1,...,k
#     x_(t,d) = f_d( l_(t,j) )                    for d = 1,...,p and j = 1,...,r
#     a_(t,s) = f_s( l_(t,j) )                    for j = 1,...,r  

#     y_(t)   = f_y( l_(t+1-i,j), a_(t+1-i,u) )   for i = 1,...,h (lag h), j = 1,...,r and u = 1,...,k 

# Parameters
np.random.seed(1); gamma = 0
n = 1000; T = 30; p = 20; r = 5; h = 3; k = 2; p_N = 5; burn = 10; tau = 5
def noise(s,sd):
    return np.random.normal(loc=0, scale=sd, size=s)
def logit(x):
    return 1/(1 + np.exp(-x))

# Coefficients
ll_coefs = np.random.normal(loc=0, scale=0.5, size=(r,h)) 
la_coefs = np.empty(shape=(r,k,h))
for i in range(h):
    for j in range(r):
        la_coefs[j,:,i] = np.random.normal(loc=1-((h-i)/h), scale=(1/h)**2, size=k)
        
xl_coefs = np.random.normal(loc=0, scale=1, size=(p,r))

ya_coefs = np.empty(shape=(1,k,h))
for i in range(h):
    ya_coefs[0,:,i] = np.random.normal(loc=1-((h-i)/h), scale=(1/h)**2, size=k) 
   
# Initialize 
L = np.random.normal(loc=0, scale=0.1, size=(n,r,T+h+burn))
X = np.random.normal(loc=0, scale=0.1, size=(n,p,T+h+burn))
A = np.round(np.random.random(size=(n,k,T+h+burn)))
Y = np.random.normal(loc=0, scale=0.1, size=(n,T+h+burn))

L_cf = np.empty(shape=(n,r,T+h+burn))
X_cf = np.empty(shape=(n,p,T+h+burn))
A_cf = np.empty(shape=(n,k,T+h+burn))
Y_cf = np.empty(shape=(n,T+h+burn))

# Simulate data with potential outcomes

po_error_l = noise((n,tau,r), 0.1)
po_error_x = noise((n,tau,p), p_N)

for pat in range(n):
        
    for j in range(h,T+h+burn-tau):    
    
        for i in range(r):
            L[pat, i, j] = (1/h)*np.dot(ll_coefs[i,:], L[pat,i,(j-h):j]) + noise(1,0.1)
            for l in range(k):
                L[pat, i, j] += (1/h)*np.dot(la_coefs[i,l], A[pat,l,(j-h):j])
    
        for i in range(p):
            X[pat, i, j] = np.dot(xl_coefs[i,:], L[pat,:,j]) + noise(1,p_N)
    
        for i in range(k):
            prob = (1-gamma)*(1/h)*np.mean(A[pat,:,j-1])
            prob += gamma*np.mean(L[pat,:,j]) 
            prob = logit(prob)
            if prob > np.random.random():
                A[pat, i, j] = 1
        
        Y[pat, j] += gamma*np.mean(L[pat,:,j]) 
        for l in range(k):
            Y[pat, j] += (1-gamma)*(1/h)*np.dot(ya_coefs[0,l], A[pat,l,(j-h+1):(j+1)])
        
    for j in range(T+h+burn-tau,T+h+burn):    
    
        for i in range(r):
            L[pat, i, j] = (1/h)*np.dot(ll_coefs[i,:], L[pat,i,(j-h):j]) + po_error_l[pat,j-(T+h+burn-tau),i]
            for l in range(k):
                L[pat, i, j] += (1/h)*np.dot(la_coefs[i,l], A[pat,l,(j-h):j])
    
        for i in range(p):
            X[pat, i, j] = np.dot(xl_coefs[i,:], L[pat,:,j]) + po_error_x[pat,j-(T+h+burn-tau),i]
    
        for i in range(k):
            prob = (1-gamma)*(1/h)*np.mean(A[pat,:,j-1]) 
            prob += gamma*np.mean(L[pat,:,j]) 
            prob = logit(prob)
            if prob > np.random.random():
                A[pat, i, j] = 1
        
        Y[pat, j] += gamma*np.mean(L[pat,:,j]) 
        for l in range(k):
            Y[pat, j] += (1-gamma)*(1/h)*np.dot(ya_coefs[0,l], A[pat,l,(j-h+1):(j+1)])

    Y_cf[pat,:] = Y[pat,:]
    L_cf[pat,:,:] = L[pat,:,:]
    X_cf[pat,:,:] = X[pat,:,:]
    A_cf[pat,:,:] = A[pat,:,:]

    for j in range(T+h+burn-tau,T+h+burn):    
    
        for i in range(r):
            L_cf[pat, i, j] = (1/h)*np.dot(ll_coefs[i,:], L_cf[pat,i,(j-h):j]) + po_error_l[pat,j-(T+h+burn-tau),i]
            for l in range(k):
                L_cf[pat, i, j] += (1/h)*np.dot(la_coefs[i,l], A_cf[pat,l,(j-h):j])
        
        for i in range(p):
                X_cf[pat, i, j] = np.dot(xl_coefs[i,:], L_cf[pat,:,j]) + po_error_x[pat,j-(T+h+burn-tau),i]
    
        for i in range(k):
            A_cf[pat, i, j] = np.round(np.random.random())
        
        Y_cf[pat, j] += gamma*np.mean(L_cf[pat,:,j]) 
        for l in range(k):
            Y_cf[pat, j] += (1-gamma)*(1/h)*np.dot(ya_coefs[0,l], A_cf[pat,l,(j-h+1):(j+1)]) 

# Data observed
data = np.empty(shape=(n,T,r+p+k+1))
for i in range(n):
    L_data = L[i,:,(h+burn):]; L_data = L_data.transpose()
    X_data = X[i,:,(h+burn):]; X_data = X_data.transpose()
    A_data = A[i,:,(h+burn):]; A_data = A_data.transpose()
    Y_data = Y[i,(h+burn):].reshape((T,1))
    data[i,:,:] = np.concatenate([Y_data, A_data, L_data, X_data], axis=1)

# Data counterfactual
data_cf = np.empty(shape=(n,T,r+p+k+1))
for i in range(n):
    L_data = L_cf[i,:,(h+burn):]; L_data = L_data.transpose()
    X_data = X_cf[i,:,(h+burn):]; X_data = X_data.transpose()
    A_data = A_cf[i,:,(h+burn):]; A_data = A_data.transpose()
    Y_data = Y_cf[i,(h+burn):].reshape((T,1))
    data_cf[i,:,:] = np.concatenate([Y_data, A_data, L_data, X_data], axis=1)

# Data split observed and counterfactual
d_train, d_test = train_test_split(data, test_size=0.1, shuffle=False)

# Data pre-processing counterfactual
d_train_cf, d_test_cf = train_test_split(data_cf, test_size=0.1, shuffle=False)

#%% Deconfounding temporal autoencoder

# Method description:
# LSTM autoencoder for time series data - inferring hidden confounders:
# Loss = Reconstruction: X replication, Outcome: Y prediction, Ignorability: Y[a] ind A | L 

# Device configuration
device = torch.device('cuda')

# Hyperparameters
rank = [0.75*p, 0.5*p, 0.25*p, 0.1*p]; drop = [0, 0.1, 0.2, 0.3]; n_layers = [1, 2, 3]
lr = [0.01, 0.005, 0.001]; alpha = [0, 0.5, 1, 2, 5]; theta = [0, 0.5, 1, 2, 5]
n_epochs = [100, 200]; b_size = [64, 128, 256]
rank = math.ceil(rank[2]); drop = drop[3]; n_layers = n_layers[1]; lr = lr[2]; alpha = alpha[1]; theta = theta[2]
n_epochs = n_epochs[0]; b_size = b_size[1]

# Data pre-processing 
train, test = train_test_split(data, test_size=0.1, shuffle=False)
train, test = torch.from_numpy(train.astype(np.float32)), torch.from_numpy(test.astype(np.float32))
train_loader = DataLoader(dataset=train, batch_size=b_size, shuffle=True)

# Potential outcomes for treatment
a_po = np.empty(shape=(2**k,train.shape[0],T,k))
def all_comb(length):
    return np.array(np.meshgrid(*[[0,1]]*length, indexing='ij')).reshape((length,-1)).transpose()
a_comb = all_comb(k)
for i in range(2**k):
    for j in range(train.shape[0]):
        for t in range(T):
            a_po[i,j,t,:] = a_comb[i]
a_po = torch.from_numpy(a_po.astype(np.float32))

# Model 
class Encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size1, embed_size, num_layers):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size1 = hidden_size1
        self.lstm = nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True, dropout=drop)
        # -> x needs to be: (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size1, embed_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(device)         
        # x: (batch_size, seq_len, input_size), h0: (num_layers, batch_size, hidden_size1)

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0,c0))  
        # out: tensor of shape (batch_size, seq_length, hidden_size1) - Outputs are hidden states
        
        out = self.fc(out)
        # out: tensor of shape (batch_size, seq_length, embed_size) 

        return out  
    
class Decoder(nn.Module):
    
    def __init__(self, embed_size, hidden_size2, x_size, a_size, y_size, num_layers):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size2 = hidden_size2
        self.lstm = nn.LSTM(embed_size, hidden_size2, num_layers, batch_first=True, dropout=drop)        
        self.fcx = nn.Linear(hidden_size2, x_size)
        self.fcy = nn.Linear(hidden_size2 + a_size, y_size)

    def forward(self, x, a):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size2).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size2).to(device)         
        # x: (batch_size, seq_len, embed_size), h0: (num_layers, batch_size, hidden_size2)

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0,c0))  
        # out: tensor of shape (batch_size, seq_length, hidden_size2) - Outputs are hidden states
        
        x_out = self.fcx(out)
        # out: tensor of shape (batch_size, seq_length, output_size) 

        out = torch.cat((out, a), dim=2)
        y_out = self.fcy(out)
        
        return x_out, y_out  

class DTA(nn.Module):
   
    def __init__(self, input_size, hidden_size1, embed_size, hidden_size2, x_size, a_size, y_size, num_layers):
        super(DTA, self).__init__()
        self.encoder = Encoder(input_size, hidden_size1, embed_size, num_layers).to(device)
        self.decoder = Decoder(embed_size, hidden_size2, x_size, a_size, y_size, num_layers).to(device)
  
    def forward(self, x, a, a_po):
        conf = self.encoder(x)
        x_out, y_out = self.decoder(conf, a)
        _, y_po_out = self.decoder(conf, a_po)
        return x_out, y_out, y_po_out, conf

mod_DTA = DTA(input_size=p, hidden_size1=math.ceil((rank+p)/2), embed_size=rank,
              hidden_size2=math.ceil((rank+p)/2), x_size=p, a_size=k, y_size=1, num_layers=n_layers).to(device)

def obj1(x_data, y_data, x_pred, y_pred, theta):
    x_loss = nn.MSELoss()(x_pred, x_data)
    y_loss = nn.MSELoss()(y_pred, y_data)
    loss = theta*x_loss + y_loss
    return loss

def obj2(y_po_data, a_data, a_data_lag, l_data, l_data_lag, weights1, weights2, alpha):
    T = y_po_data.shape[1]
    mat1 = torch.cat((a_data,a_data_lag,l_data,l_data_lag), dim=2)
    mat2 = torch.cat((l_data,l_data_lag), dim=2)
    mu1 = torch.sum(mat1*weights1, dim=2)
    mu2 = torch.sum(mat2*weights2, dim=2)
    sd1 = torch.sqrt(torch.mean((y_po_data-mu1)**2, dim=0))
    sd2 = torch.sqrt(torch.mean((y_po_data-mu2)**2, dim=0))
    KL_dist = (1/T)*torch.sum(torch.log(sd2/sd1) + (sd1**2 + torch.mean((mu1-mu2)**2, dim=0))/(2*sd2**2) - 1/2)
    return alpha*KL_dist    
    
optimizer = torch.optim.Adam(mod_DTA.parameters(), lr=lr)  

# Train the model
for epoch in range(n_epochs):
    for batch in train_loader: 
        y_data = batch[:,:,0:1]
        a_data = batch[:,:,1:(1+k)]
        x_data = batch[:,:,(1+k+r):(1+k+r+p)]
        y_data = y_data.to(device)
        a_data = a_data.to(device)
        x_data = x_data.to(device)
        # Forward pass
        x_out, y_out, y_po_out, conf = mod_DTA(x_data, a_data, a_data)
        loss_x = obj1(x_data, y_data, x_out, y_out, theta=theta)
        # Backward and optimize
        loss_x.backward()
        optimizer.step()
        optimizer.zero_grad()
    for v in range(2**k):
        y_data = train[:,:,0:1]
        a_data = train[:,:,1:(1+k)]
        x_data = train[:,:,(1+k+r):(1+k+r+p)]
        a_data_po = a_po[v,:,:,:]
        y_data = y_data.to(device)
        a_data = a_data.to(device)
        x_data = x_data.to(device)
        a_data_po = a_data_po.to(device)
        # Forward pass
        x_out, y_out, y_po_out, conf = mod_DTA(x_data, a_data, a_data_po)
        y_po_out = y_data - y_out + y_po_out
        w1 = torch.empty(T-1,2*(rank+k))
        w2 = torch.empty(T-1,2*rank)
        for t in range(1,T):
            mod_linr = LinearRegression()
            Y = y_po_out[:,t,:].detach()
            A = a_data[:,t,:]
            A1 = a_data[:,t-1,:]
            mat1 = torch.cat((A,  A1, conf[:,t,:].detach(), conf[:,t-1,:].detach()), dim=1)
            mat2 = torch.cat((conf[:,t,:].detach(), conf[:,t-1,:].detach()), dim=1)
            w1[t-1,:] = torch.tensor(mod_linr.fit(mat1.cpu().detach().numpy(),Y.cpu().numpy()).coef_)
            w2[t-1,:] = torch.tensor(mod_linr.fit(mat2.cpu().detach().numpy(),Y.cpu().numpy()).coef_)
        Y = y_po_out[:,1:T,0].detach()
        A = a_data[:,1:T,:]
        Alag = a_data[:,0:(T-1),:]
        L = conf[:,1:T,:]
        Llag = conf[:,0:(T-1),:]
        w1 = w1.to(device)
        w2 = w2.to(device)
        loss_kl = obj2(Y, A, Alag, L, Llag, w1, w2, alpha=alpha)
        # Backward and optimize
        loss_kl.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1) % (n_epochs/100) == 0:
        print (f'Epoch [{epoch+1}/{n_epochs}], Loss_x: {loss_x.item():.4f}, Loss_kl: {loss_kl.item():.4f}')

# Predict confounders with trained model
train = train.to(device)
test = test.to(device)
_, _, _, conf_DTA = mod_DTA(train[:,:,(1+k+r):(1+k+r+p)], train[:,:,1:(1+k)], 1-train[:,:,1:(1+k)])
_, _, _, conf_DTA_test = mod_DTA(test[:,0:(T-tau+1),(1+k+r):(1+k+r+p)], test[:,0:(T-tau+1),1:(1+k)], 
                                 test[:,0:(T-tau+1),1:(1+k)])
conf_DTA = conf_DTA.cpu().detach().numpy()
conf_DTA_test = conf_DTA_test.cpu().detach().numpy()

#%% Marginal Structural Models (Benchmark 1)

# Description: continuous outcome, multivariate binary treatment, continuous or binary covariates
# Data is of shape (n,T,d), where d=1 for outcome, k for treatment, (r or p) for time covariates and s for static features

def get_SW(Y, A, L, tau):
    
    n,T,k = A.shape
    p = L.shape[2]
    
    # Marginal probability of individual treatment given treatment history (Logistic regression)
    fm_At = np.empty(shape=(n,tau,2*k))        
    for i in range(k):        
        for t in range(T-tau,T):
            y = A[:,t,i]
            X = np.empty(shape=(n,k))
            for m in range(k):
                X[:,m] = np.sum(A[:,0:t,m], axis=1)
            sc_X = StandardScaler()
            X = sc_X.fit_transform(X)
            mod_logr = LogisticRegression(penalty='none', max_iter=200)
            mod_logr.fit(X,y)
            fm_At[:,t-(T-tau),(2*i):(2*i+2)] = mod_logr.predict_proba(X)   
    
    # Marginal probability of individual treatment given patient history
    fm_AtHt = np.empty(shape=(n,tau,2*k))
    for i in range(k):
        for t in range(T-tau,T):
            y = A[:,t,i]
            X = np.empty(shape=(n,k+2*p+1)) 
            for m in range(k):
                X[:,m] = np.sum(A[:,0:t,m], axis=1)
            for m in range(p):
                X[:,(k+m)] = L[:,t-1,m]
            for m in range(p):
                X[:,(k+p+m)] = L[:,t,m]
            X[:,(k+2*p)] = Y[:,t-1]
            sc_X = StandardScaler()
            X = sc_X.fit_transform(X)
            mod_logr = LogisticRegression(penalty='none', max_iter=200)
            mod_logr.fit(X,y)
            fm_AtHt[:,t-(T-tau),(2*i):(2*i+2)] = mod_logr.predict_proba(X)      
    
    # Joint probability of treatment given treatment history (product of marginals)
    fm_A = np.ones(shape=(n,tau))
    for i in range(n):
        for t in range(T-tau,T):
            for j in range(k):
                if A[i,t,j] == 0:
                    fm_A[i,t-(T-tau)] *= fm_At[i,t-(T-tau),2*j]
                else:
                    fm_A[i,t-(T-tau)] *= fm_At[i,t-(T-tau),2*j+1]
            
    # Joint probability of treatment given patient history (product of marginals)
    fm_AH = np.ones(shape=(n,tau))
    for i in range(n):
        for t in range(T-tau,T):
            for j in range(k):
                if A[i,t,j] == 0:
                    fm_AH[i,t-(T-tau)] *= fm_AtHt[i,t-(T-tau),2*j]
                else:
                    fm_AH[i,t-(T-tau)] *= fm_AtHt[i,t-(T-tau),2*j+1]
    
    # Stabilized IP weights for each time-step from t,...,T (T = t + tau)
    SW_tau = fm_A/fm_AH
    
    # Cumulative stabilized weights for period t,...,T (product of weights for given period)
    SW_t = np.ones(n)
    for i in range(n):
        for t in range(tau):
            SW_t[i] *= SW_tau[i,t]
    
    return SW_t

# MSM comparisons observed and counterfactual
s_ahead = tau

# MSM 
sw = get_SW(Y=d_train[:,:,0], A=d_train[:,:,1:(1+k)], 
            L=d_train[:,:,(1+k+r):(1+k+r+p)], tau=tau)  
mod_linr = LinearRegression()
y_train = d_train[:,(T-1+s_ahead-tau),0]
X_train = np.empty(shape=(d_train.shape[0],2*k+2*p+1))
for i in range(k):
    X_train[:,i] = np.sum(d_train[:,(T-tau):(T+s_ahead-tau),1+i], axis=1)
for i in range(k):
    X_train[:,k+i] = np.sum(d_train[:,0:(T-tau),1+i], axis=1)
for i in range(p):
    X_train[:,2*k+i] = d_train[:,T-tau,1+k+r+i]
for i in range(p):
    X_train[:,2*k+p+i] = d_train[:,T-tau-1,1+k+r+i]
X_train[:,(2*k+2*p)] = d_train[:,T-tau-1,0]
mod_linr.fit(X_train, y_train, sample_weight=sw)

y_test = d_test_cf[:,(T-1+s_ahead-tau),0]   
X_test = np.empty(shape=(d_test_cf.shape[0],2*k+2*p+1))
for i in range(k):
    X_test[:,i] = np.sum(d_test_cf[:,(T-tau):(T+s_ahead-tau),1+i], axis=1)
for i in range(k):
    X_test[:,k+i] = np.sum(d_test_cf[:,0:(T-tau),1+i], axis=1)
for i in range(p):
    X_test[:,2*k+i] = d_test_cf[:,T-tau,1+k+r+i]
for i in range(p):
    X_test[:,2*k+p+i] = d_test_cf[:,T-tau-1,1+k+r+i]
X_test[:,(2*k+2*p)] = d_test_cf[:,T-tau-1,0]
y_pred = mod_linr.predict(X_test)
loss1_cf = np.sqrt(np.mean((y_pred-y_test)**2))
print('Loss MSM: ', loss1_cf)

# MSM + DTA
sw = get_SW(Y=d_train[:,:,0], A=d_train[:,:,1:(1+k)], 
            L=conf_DTA, tau=tau)  
mod_linr = LinearRegression()
y_train = d_train[:,(T-1+s_ahead-tau),0]
X_train = np.empty(shape=(d_train.shape[0],2*k+2*rank+1))
for i in range(k):
    X_train[:,i] = np.sum(d_train[:,(T-tau):(T+s_ahead-tau),1+i], axis=1)
for i in range(k):
    X_train[:,k+i] = np.sum(d_train[:,0:(T-tau),1+i], axis=1)
for i in range(rank):
    X_train[:,2*k+i] = conf_DTA[:,T-tau,i]
for i in range(rank):
    X_train[:,2*k+rank+i] = conf_DTA[:,T-tau-1,i]
X_train[:,(2*k+2*rank)] = d_train[:,T-tau-1,0]
mod_linr.fit(X_train, y_train, sample_weight=sw)

y_test = d_test_cf[:,(T-1+s_ahead-tau),0]   
X_test = np.empty(shape=(d_test_cf.shape[0],2*k+2*rank+1))
for i in range(k):
    X_test[:,i] = np.sum(d_test_cf[:,(T-tau):(T+s_ahead-tau),1+i], axis=1)
for i in range(k):
    X_test[:,k+i] = np.sum(d_test_cf[:,0:(T-tau),1+i], axis=1)
for i in range(rank):
    X_test[:,2*k+i] = conf_DTA_test[:,T-tau,i]
for i in range(rank):
    X_test[:,2*k+rank+i] = conf_DTA_test[:,T-tau-1,i]
X_test[:,(2*k+2*rank)] = d_test_cf[:,T-tau-1,0]
y_pred = mod_linr.predict(X_test)
loss2_cf = np.sqrt(np.mean((y_pred-y_test)**2))
print('Loss DTA: ', loss2_cf)

# MSM oracle
sw = get_SW(Y=d_train[:,:,0], A=d_train[:,:,1:(1+k)], 
            L=d_train[:,:,(1+k):(1+k+r)], tau=tau)  
mod_linr = LinearRegression()
y_train = d_train[:,(T-1+s_ahead-tau),0]
X_train = np.empty(shape=(d_train.shape[0],2*k+2*r+1))
for i in range(k):
    X_train[:,i] = np.sum(d_train[:,(T-tau):(T+s_ahead-tau),1+i], axis=1)
for i in range(k):
    X_train[:,k+i] = np.sum(d_train[:,0:(T-tau),1+i], axis=1)
for i in range(r):
    X_train[:,2*k+i] = d_train[:,T-tau,1+k+i]
for i in range(r):
    X_train[:,2*k+r+i] = d_train[:,T-tau-1,1+k+i]
X_train[:,(2*k+2*r)] = d_train[:,T-tau-1,0]
mod_linr.fit(X_train, y_train, sample_weight=sw)

y_test = d_test_cf[:,(T-1+s_ahead-tau),0]   
X_test = np.empty(shape=(d_test_cf.shape[0],2*k+2*r+1))
for i in range(k):
    X_test[:,i] = np.sum(d_test_cf[:,(T-tau):(T+s_ahead-tau),1+i], axis=1)
for i in range(k):
    X_test[:,k+i] = np.sum(d_test_cf[:,0:(T-tau),1+i], axis=1)
for i in range(r):
    X_test[:,2*k+i] = d_test_cf[:,T-tau,1+k+i]
for i in range(r):
    X_test[:,2*k+r+i] = d_test_cf[:,T-tau-1,1+k+i]
X_test[:,(2*k+2*r)] = d_test_cf[:,T-tau-1,0]
y_pred = mod_linr.predict(X_test)
loss3_cf = np.sqrt(np.mean((y_pred-y_test)**2))
print('Loss MSM oracle: ', loss3_cf)

#%% Recurrent Marginal Structural Networks (Benchmark 2)

# Description: continuous outcome, binary treatment, continuous or binary covariates
# Data is of shape (n,T,d), where d=1 for outcome, k for treatment, p for time covariates 
# Procedure: estimate IPTW and re-weight data, estimate encoder, and estimate decoder with pre-trained encoder

def RMSN(Y_train, A_train, L_train, tau, Y_test, A_test, L_test):
    
    n,T,k = A_train.shape
    p = L_train.shape[2]
    
    # Data for estimating propensity weights (treatment classification)
    data_train = np.concatenate([Y_train, A_train, L_train], axis=2)
    data_pw = data_train
    a_class = np.empty(shape=(n,T,1))
    data_pw = np.concatenate([data_pw, a_class], axis=2)

    for pat in range(n):
        for t in range(T):
            for v in range(2**k):
                if np.array_equal(data_pw[pat,t,1:(1+k)], a_comb[v]):
                    data_pw[pat,t,1+k+p] = v

    data_pw = torch.from_numpy(data_pw.astype(np.float32))
    train_loader_pw = DataLoader(dataset=data_pw, batch_size=math.ceil(n/20), shuffle=True)

    # Propensity network
    # Description: estimating stabilized propensity weights by predicting treatment using LSTM architecture

    # Device configuration
    device = torch.device('cuda')

    class PropNet(nn.Module):

        def __init__(self, input_size, hidden_size, num_classes, num_layers):
            super(PropNet, self).__init__()
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            # -> x needs to be: (batch_size, seq_length, input_size)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)         
            # x: (batch_size, seq_len, input_size), h0: (num_layers, batch_size, hidden_size1)

            # Forward propagate RNN
            out, _ = self.lstm(x, (h0,c0))  
            # out: tensor of shape (batch_size, seq_length, hidden_size1) - Outputs are hidden states

            out = self.fc(out)
            # out: tensor of shape (batch_size, seq_length, num_classes)

            return out  

    # Estimate numerator
    mod_PN_num = PropNet(input_size=k, hidden_size=k, num_classes=2**k, num_layers=1).to(device)
    obj = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mod_PN_num.parameters(), lr=0.001)  

    # Train 
    n_epochs = 100
    for epoch in range(n_epochs):
        for batch in train_loader_pw: 
            y_data = batch[:,1:T,(1+k+p)]
            y_data = y_data.long()
            x_data = batch[:,0:(T-1),1:(1+k)]
            y_data = y_data.to(device)
            x_data = x_data.to(device)
            # Forward pass
            out = mod_PN_num(x_data)
            out = torch.transpose(out,1,2)
            loss = obj(out, y_data)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (epoch+1) % (n_epochs) == 0:
            print (f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    # Estimate denominator
    mod_PN_den = PropNet(input_size=1+p+k, hidden_size=1+p+k, num_classes=2**k, num_layers=1).to(device)
    obj = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mod_PN_den.parameters(), lr=0.001)  

    # Train 
    n_epochs = 100
    for epoch in range(n_epochs):
        for batch in train_loader_pw: 
            y_data = batch[:,1:T,(1+k+p)]
            y_data = y_data.long()
            x_data = torch.cat((batch[:,0:(T-1),0:1], batch[:,0:(T-1),1:(1+k)],
                                batch[:,1:T,(1+k):(1+k+p)]), dim=2)
            y_data = y_data.to(device)
            x_data = x_data.to(device)
            # Forward pass
            out = mod_PN_den(x_data)
            out = torch.transpose(out,1,2)
            loss = obj(out, y_data)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (epoch+1) % (n_epochs) == 0:
            print (f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    # Calculate weights
    data_pw = data_pw.to(device)

    fm_At = mod_PN_num(data_pw[:,0:(T-1),1:(1+k)])
    fm_At = nn.Softmax(dim=2)(fm_At)
    fm_At = fm_At.cpu().detach().numpy()
    fm_A = np.ones(shape=(data_pw.shape[0],tau))
    for i in range(data_pw.shape[0]):
        for t in range(T-tau,T):
            for v in range(2**k):
                if np.array_equal(data_train[i,t,1:(1+k)], a_comb[v]):
                    fm_A[i,t-(T-tau)] = fm_At[i,t-1,v]    

    fm_AtHt = mod_PN_den(torch.cat((data_pw[:,0:(T-1),0:1], data_pw[:,0:(T-1),1:(1+k)],
                                    data_pw[:,1:T,(1+k):(1+k+p)]), dim=2))
    fm_AtHt = nn.Softmax(dim=2)(fm_AtHt)
    fm_AtHt = fm_AtHt.cpu().detach().numpy()
    fm_AH = np.ones(shape=(data_pw.shape[0],tau))
    for i in range(data_pw.shape[0]):
        for t in range(T-tau,T):
            for v in range(2**k):
                if np.array_equal(data_train[i,t,1:(1+k)], a_comb[v]):
                    fm_AH[i,t-(T-tau)] = fm_AtHt[i,t-1,v]                    

    SW_tau = fm_A/fm_AH
    SW_t = np.ones(data_pw.shape[0])
    for i in range(data_pw.shape[0]):
        for t in range(tau):
            SW_t[i] *= SW_tau[i,t]
            
    # Data for estimating encoder (weights added to data)
    data_en = data_train
    SW_rmsn = torch.from_numpy(SW_t.astype(np.float32))
    SW_rmsn = SW_rmsn.to(device)

    data_en = torch.from_numpy(data_en.astype(np.float32))
    train_loader_en = DataLoader(dataset=data_en, batch_size=math.ceil(n/20), shuffle=False)
    
    # Encoder
    # Description: estimating one-step-ahead prediction to build good representation of patient history

    class Encoder(nn.Module):

        def __init__(self, input_size, hidden_size, y_size, num_layers):
            super(Encoder, self).__init__()
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            # -> x needs to be: (batch_size, seq_length, input_size)
            self.fc = nn.Linear(hidden_size, y_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)         
            # x: (batch_size, seq_len, input_size), h0: (num_layers, batch_size, hidden_size1)

            # Forward propagate RNN
            out, _ = self.lstm(x, (h0,c0))  
            # out: tensor of shape (batch_size, seq_length, hidden_size) - Outputs are hidden states

            out = out[:, -1, :]
            # out: tensor of shape (batch_size, hidden_size) - Outputs are hidden states

            h_out = out
            y_out = self.fc(out)
            # out: tensor of shape (batch_size, y_size)

            return y_out, h_out  

    mod_RMSN_en = Encoder(input_size=1+p+k, hidden_size=1+p+k, y_size=1, num_layers=1).to(device)

    def obj(y_pred, y_true, w):
        loss = torch.mean(w*(y_pred-y_true)**2)
        return loss

    optimizer = torch.optim.Adam(mod_RMSN_en.parameters(), lr=0.001)  

    # Train 
    n_epochs = 100
    for epoch in range(n_epochs):
        for i,batch in enumerate(train_loader_en): 
            y_data = batch[:,T-tau-1,0]
            x_data = torch.cat((torch.cat((torch.zeros(batch.size(0),1,1), batch[:,0:(T-tau-1),0:1]), dim=1), 
                                batch[:,0:(T-tau),1:(1+k)], batch[:,0:(T-tau),(1+k):(1+k+p)]), dim=2)
            y_data = y_data.to(device)
            x_data = x_data.to(device)
            # Forward pass
            out, _ = mod_RMSN_en(x_data)
            loss = obj(out[:,0], y_data, SW_rmsn[(i*batch.size(0)):((i+1)*batch.size(0))])
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (epoch+1) % (n_epochs) == 0:
            print (f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    # Decoder
    # Description: predicting Y using the treatment policy and history representation from encoder network

    class Decoder(nn.Module):

        def __init__(self, input_size, hidden_size, y_size, num_layers, tau):
            super(Decoder, self).__init__()
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            # -> x needs to be: (batch_size, seq_length, input_size)
            self.fc = nn.Linear(hidden_size, y_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)         
            # x: (batch_size, seq_len, input_size), h0: (num_layers, batch_size, hidden_size1)

            # Forward propagate RNN
            out, _ = self.lstm(x, (h0,c0))  
            # out: tensor of shape (batch_size, seq_length, hidden_size) - Outputs are hidden states

            y_out = self.fc(out)
            # out: tensor of shape (batch_size, seq_length, y_size)

            return y_out  

    mod_RMSN_de = Decoder(input_size=1+p+2*k, hidden_size=1+p+2*k, y_size=1, tau = tau, num_layers=1).to(device)

    def obj(y_pred, y_true, w):
        loss = torch.mean(w*torch.mean((y_pred-y_true)**2, dim=1))
        return loss

    optimizer = torch.optim.Adam(mod_RMSN_de.parameters(), lr=0.001)  

    # Train 
    n_epochs = 100
    for epoch in range(n_epochs):
        for i,batch in enumerate(train_loader_en): 
            y_data = batch[:,(T-tau):T,0]
            y_data = y_data.to(device)
            x1_data = torch.cat((torch.cat((torch.zeros(batch.size(0),1,1), batch[:,0:(T-tau-1),0:1]), dim=1), 
                                 batch[:,0:(T-tau),1:(1+k)], batch[:,0:(T-tau),(1+k):(1+k+p)]), dim=2)
            x1_data = x1_data.to(device)
            _, out = mod_RMSN_en(x1_data)
            adapter = torch.empty(batch.size(0), tau, out.size(1))
            for j in range(batch.size(0)):
                for t in range(tau):
                    adapter[j,t] = out[j]            
            x_data = torch.cat((adapter, batch[:,(T-tau):T,1:(1+k)]), dim=2)
            x_data = x_data.to(device)
            # Forward pass
            out = mod_RMSN_de(x_data)
            loss = obj(out[:,:,0], y_data, SW_rmsn[(i*batch.size(0)):((i+1)*batch.size(0))])
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (epoch+1) % (n_epochs) == 0:
            print (f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
    
    # Predict with the trained models
    Y_test, A_test = torch.from_numpy(Y_test.astype(np.float32)), torch.from_numpy(A_test.astype(np.float32))
    L_test = torch.from_numpy(L_test.astype(np.float32))
    
    x_en = torch.cat((torch.cat((torch.zeros(Y_test.size(0),1,1), Y_test[:,0:(T-tau-1),:]), dim=1), 
                      A_test[:,0:(T-tau),:], L_test[:,0:(T-tau),:]), dim=2)
    x_en = x_en.to(device)
    _, out = mod_RMSN_en(x_en)
    adapt = torch.empty(Y_test.size(0), tau, out.size(1))
    for j in range(Y_test.size(0)):
        for t in range(tau):
            adapt[j,t] = out[j] 
    
    x_de = torch.cat((adapt, A_test[:,(T-tau):T,:]), dim=2)
    x_de = x_de.to(device)
    y_pred = mod_RMSN_de(x_de)
    
    return y_pred.cpu().detach().numpy()

# RMSN comparisons observed and counterfactual

s_ahead = tau
# RMSN
y_pred = RMSN(Y_train=d_train[:,:,0:1], A_train=d_train[:,:,1:(1+k)], 
              L_train=d_train[:,:,(1+k+r):(1+k+r+p)], tau=tau,
              Y_test=d_test_cf[:,:,0:1], A_test=d_test_cf[:,:,1:(1+k)], 
              L_test=d_test_cf[:,:,(1+k+r):(1+k+r+p)])
y_pred = y_pred[:, s_ahead-1, 0]
y_test = d_test_cf[:,(T-1+s_ahead-tau),0]   
loss1_cf = np.sqrt(np.mean((y_pred-y_test)**2))
print('Loss RMSN: ', loss1_cf)

# RMSN + DTA
y_pred = RMSN(Y_train=d_train[:,:,0:1], A_train=d_train[:,:,1:(1+k)], 
              L_train=conf_DTA, tau=tau, Y_test=d_test_cf[:,:,0:1], 
              A_test=d_test_cf[:,:,1:(1+k)], L_test=conf_DTA_test)
y_pred = y_pred[:, s_ahead-1, 0]
y_test = d_test_cf[:,(T-1+s_ahead-tau),0]   
loss2_cf = np.sqrt(np.mean((y_pred-y_test)**2))
print('Loss DTA: ', loss2_cf)

# RMSN oracle
y_pred = RMSN(Y_train=d_train[:,:,0:1], A_train=d_train[:,:,1:(1+k)], 
              L_train=d_train[:,:,(1+k):(1+k+r)], tau=tau,
              Y_test=d_test_cf[:,:,0:1], A_test=d_test_cf[:,:,1:(1+k)], 
              L_test=d_test_cf[:,:,(1+k):(1+k+r)])
y_pred = y_pred[:, s_ahead-1, 0]
y_test = d_test_cf[:,(T-1+s_ahead-tau),0]   
loss3_cf = np.sqrt(np.mean((y_pred-y_test)**2))
print('Loss RMSN oracle: ', loss3_cf)
