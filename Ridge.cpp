#include <Rcpp.h>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// Fonction pour effectuer la régression ridge étape par étape
// [[Rcpp::export]]
List regressionRidge(const arma::mat& X, const arma::vec& y, double lambda) {
  int n = X.n_rows; // Nombre d'échantillons
  int p = X.n_cols; // Nombre de variables indépendantes

  // Étape 1 : Standardisation des données
  arma::mat X_std = arma::normalise(X);

  // Étape 2 : Calcul de la matrice de covariance
  arma::mat cov_matrix = (X_std.t() * X_std) / n;

  // Étape 3 : Ajout de la pénalité
  arma::mat ridge_matrix = cov_matrix + lambda * arma::eye<arma::mat>(p, p);

  // Étape 4 : Calcul des coefficients
  arma::vec coefficients = arma::solve(ridge_matrix, X_std.t() * y);

  // Étape 5 : Prédiction des valeurs
  arma::vec predictions = X_std * coefficients;

  // Retourne les résultats sous forme de liste
  return List::create(Named("coefficients") = coefficients,
                      Named("predictions") = predictions);
}