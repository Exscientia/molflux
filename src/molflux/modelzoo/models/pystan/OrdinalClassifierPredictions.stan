/*
see eg. https://medium.com/@alex.pavlakis/making-predictions-from-stan-models-in-r-3e349dfac1ed
*/
data {
  int<lower=3> n_classes;
  int<lower=10> N_samples;
  int<lower=1> N_pred;         // number of compounds
  int<lower=1> P;         // number of coefs
  matrix[N_pred,P] X_pred;          // assay and other data
  array[N_samples] ordered[n_classes - 1] cutpoints;      // n_classes-1 thresholds
  matrix[N_samples, P] beta;            // parameters
}
parameters {
}
model {
}
generated quantities{
  matrix[N_samples, N_pred] ypred;     // predict category
  matrix[N_samples, N_pred] eta_pred;  // eta for new data

  // new compounds
  for (i in 1:N_samples){
    eta_pred[i] = to_row_vector(X_pred * to_vector(beta[i]));
    for (j in 1:N_pred) {
      ypred[i, j] = ordered_logistic_rng(eta_pred[i, j], cutpoints[i]);
    }
  }
}