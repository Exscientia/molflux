/*
Shrinkage parameter (sigma) is learned from the data
*/

data{
  int<lower=1> N;         // number of compounds
  int<lower=1> P;         // number of coefs
  int<lower=3> n_classes; // numer of classes
  array[N] int y;               // outcome
  matrix[N,P] X;          // assay and other data
  real sigma_prior;       // prior for sigma
  real mu_prior;          // prior for mu
}

parameters{
  ordered[n_classes - 1] cutpoints;      // n_classes-1 thresholds
  vector[P] beta;            // parameters
  real<lower=0> sigma;       // tuning param for prior over coefs
  real mu;                   // mean for coefs... should be close to zero
}

model{
  vector[N] eta;             // linear predictor

  cutpoints ~ normal(0, 20); // prior for cutpoints

  // Laplace prior on coefs (similar to L1 reg)
  sigma ~ normal(0, sigma_prior);
  mu ~ normal(0, mu_prior);
  beta ~ double_exponential(mu, sigma);

  // likelihood
  eta = X * beta;
  for (i in 1:N){
    y[i] ~ ordered_logistic(eta[i], cutpoints);
  }
}