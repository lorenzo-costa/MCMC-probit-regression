# MCMC Bayesian Probit regression
As part of a course in Probability and Statistic together with 4 collegues we tackled the problem of finding coefficients in a Bayesian Probit Regression using Metropoli Hasting MCMC (Markov Chain Monte Carlo) techniques

The file named algorithms.py contains the two algorithm we tried to implement:
- in MCMC_random at each iteration we randomly select one of the parameters to update, generate a proposal and then accept/reject it
- in MCMC_gibbs at each iteration we update all the parameters following the logic of Gibbs sampler (we condition the distribution on the latest selected parameters)
