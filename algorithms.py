import numpy as np


def MCMC_random(X, Y, beta0, V0, posterior, V=None, iterations=5000, verbose=True, decay=False):
    d = len(beta0)
    beta_sample = np.zeros((iterations, d))
    accepted = np.zeros(d)
    sampled = np.zeros(d)
    beta_sample[0, :] = beta0
        
    #Fine-tuned variances for the proposal distribution:
    if V is None:
        V = np.ones(d)*0.01
        
    for i in range(1, iterations):
        if decay is True:
            if i == round(iterations*1/2):
                V *= 0.1
            #if i == round(iterations*1/3):
            #    V *= 0.1
        temp_beta = beta_sample[i-1,:].copy()        #here beta_d will be substituded by the proposal
        condition_beta = beta_sample[i-1,:].copy()   #contains the most recent samples of beta
        
        b = np.random.choice(np.arange(d))
        sampled[b] += 1
        new_beta = np.random.normal(loc = beta_sample[i-1,b], scale = V[b], size=1)
            
        temp_beta[b] = new_beta
        beta_log_diff = posterior(temp_beta, Y, X, beta0, V0) - posterior(condition_beta, Y, X, beta0, V0)
        
        if np.log(np.random.rand()) < beta_log_diff: #ACCEPT the single beta
            temp_beta[b] = new_beta
            condition_beta[b] = new_beta
            accepted[b] += 1
        else: #REJECT
            temp_beta[b] = beta_sample[i-1,b]
            condition_beta[b] = beta_sample[i-1,b]
        
        #save the sampled beta's
        beta_sample[i,:] = condition_beta[:]
            
        
        if verbose and i%(iterations/10)==0:
            print(f"{i}/{iterations}")
            print(accepted/sampled)
    
    return beta_sample, accepted/sampled

def MCMC_random_unif(X, Y, beta0, V0, posterior, iterations=5000, verbose=True, decay=False, offset = 0.5):
    d = len(beta0)
    beta_sample = np.zeros((iterations, d))
    accepted = np.zeros(d)
    sampled = np.zeros(d)
    beta_sample[0, :] = beta0
    c = offset
    
    for i in range(1, iterations):
        if decay is True:
            if i == iterations//2:
                offset *= 0.1
        
        temp_beta = beta_sample[i-1,:].copy()        #here beta_d will be substituded by the proposal
        condition_beta = beta_sample[i-1,:].copy()   #contains the most recent samples of beta
        
        b = np.random.choice(np.arange(d))
        sampled[b] += 1
        new_beta = np.random.uniform(low = beta_sample[i-1,b]-offset[b], high = beta_sample[i-1,b]+offset[b])
            
        temp_beta[b] = new_beta
        beta_log_diff = posterior(temp_beta, Y, X, beta0, V0) - posterior(condition_beta, Y, X, beta0, V0)
        
        if np.log(np.random.rand()) < beta_log_diff: #ACCEPT the single beta
            temp_beta[b] = new_beta
            condition_beta[b] = new_beta
            accepted[b] += 1
        else: #REJECT
            temp_beta[b] = beta_sample[i-1,b]
            condition_beta[b] = beta_sample[i-1,b]
        
        #save the sampled beta's
        beta_sample[i,:] = condition_beta[:]
            
        if verbose and i%(iterations/10)==0:
            print(f"{i}/{iterations}")
            print(accepted/sampled)
    
    return beta_sample, accepted/sampled

def MCMC_metropolis_hastings(X, Y, beta0, V0, posterior, V, iterations=5000, verbose=True):
    d = len(beta0)
    beta_sample = np.zeros((iterations, d))
    accepted = np.zeros(d)
    beta_sample[0, :] = beta0
    
    posterior = posterior
   
    for i in range(1, iterations):
        #Gibbs sampler: condition on the most recent betas
        temp_beta = beta_sample[i-1,:].copy()        #here beta_d will be substituded by the proposal
        condition_beta = beta_sample[i-1,:].copy()   #contains the most recent samples of beta
        
        for b in range(d):
            new_beta = np.random.normal(loc = beta_sample[i-1,b], scale = V[b], size=1)
            
            temp_beta[b] = new_beta
            beta_log_diff = posterior(temp_beta, Y, X, beta0, V0) - posterior(condition_beta, Y, X, beta0, V0)
        
            if np.log(np.random.rand()) < beta_log_diff: #ACCEPT the single beta
                temp_beta[b] = new_beta
                condition_beta[b] = new_beta
                accepted[b] += 1
            else: #REJECT
                temp_beta[b] = beta_sample[i-1,b]
                condition_beta[b] = beta_sample[i-1,b]
        
        #save the sampled beta's
        beta_sample[i,:] = condition_beta[:]
            
        
        if verbose and i%(iterations/10)==0:
            print(f"{i}/{iterations}")
            print(accepted/i)
    
    return beta_sample, accepted/iterations