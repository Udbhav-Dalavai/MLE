library(igraph)
library(mclust)


n <- 100
k <- 2    
p <- 0.7  
q <- 0.5  


block_sizes <- rep(n / k, k)
block_assignments <- unlist(lapply(seq_along(block_sizes), function(x) rep(x, block_sizes[x])))


block_probabilities <- matrix(c(p, q, q, p), nrow = k, ncol = k)


set.seed(142)
g <- sample_sbm(n, pref.matrix = block_probabilities, block.sizes = block_sizes, directed = FALSE)


A <- as_adjacency_matrix(g, sparse = FALSE)


A_noisy <- A
A_noisy[sample(which(A == 1), size = 0.2 * sum(A == 1))] <- 0  #Randomly set 20% of 1's to 0
A_noisy[sample(which(A == 0), size = 0.05 * sum(A == 0))] <- 1  #Add noise (5% of 0's become 1)


optimize_likelihood <- function(A, d, g = function(x) 1 / (1 + exp(-x)), 
                                tol = 1e-5, max_iter = 1000, lr = 0.001, lambda = 0.01, 
                                initial_X = NULL) {
  n <- nrow(A)
  
  #Initialize M as a mask and low-rank approximation (Y = WX)
  M <- ifelse(A == 0, 0, 1) * (1 - diag(1, n))
  
  W <- matrix(rnorm(d * d, mean = 0, sd = 0.01), d, d) 
  
  if (is.null(initial_X)) {
    X <- matrix(rnorm(n * d, mean = 0, sd = 0.01), n, d)  
  } else {
    X <- initial_X
  }
  
  clip <- function(x, eps = 1e-6) {
    pmin(pmax(x, eps), 1 - eps)
  }
  
  log_likelihood <- function(A, X, W, g, lambda) {
    P <- clip(g(X %*% t(X)))
    Y <- X %*% W
    M_approx <- Y %*% t(Y)
    A_obs <- A + M_approx
    sum((A_obs * (X %*% t(X)) + log(1 - P))) - lambda * (sum(X^2) + sum(W^2)) -
      sum((M - M_approx)^2)
  }
  
  gradient_X <- function(A, X, W, g, lambda) {
    P <- g(X %*% t(X))
    Y <- X %*% W
    M_approx <- Y %*% t(Y)
    grad_X <- 2 * (A + M_approx - ((1 - (A + M_approx)) * P)) %*% X - 4 * (M - M_approx) %*% Y %*% t(W) - 2 * lambda * X
    return(grad_X)
  }
  
  gradient_W <- function(A, X, W, M) {
    Y <- X %*% W
    M_approx <- Y %*% t(Y)
    grad_W <- -4 * t(X) %*% (M - M_approx) %*% Y - 2 * lambda * W
    return(grad_W)
  }
  
  clip_gradient <- function(grad, max_norm = 1) {
    norm <- sqrt(sum(grad^2))
    if (norm > max_norm) {
      grad <- grad * (max_norm / norm)
    }
    return(grad)
  }
  
  for (iter in 1:max_iter) {
    grad_X <- gradient_X(A, X, W, g, lambda)
    grad_W <- gradient_W(A, X, W, M)
    
    grad_X <- clip_gradient(grad_X) 
    grad_W <- clip_gradient(grad_W) 
    
    X <- X + lr * grad_X 
    W <- W + lr * grad_W 
    
    if (sqrt(sum(grad_X^2) + sum(grad_W^2)) < tol) break
  }
  
  list(X = X, W = W, log_likelihood = log_likelihood(A, X, W, g, lambda))
}


#ASE
ase_embedding <- function(A, d) {
  eig <- eigen(A, symmetric = TRUE)
  X <- eig$vectors[, 1:d] %*% diag(sqrt(eig$values[1:d]))
  return(X)
}

d <- 2
X_ase <- ase_embedding(A_noisy, d)

plot(X_ase, col = block_assignments, pch = 16, main = "ASE Embedding")

#likelihood-based optimization
result <- optimize_likelihood(A_noisy, d = d, lambda = 0.02, lr = 0.005, max_iter = 2000, initial_X = X_ase)

X_likelihood <- result$X
W_likelihood <- result$W

plot(X_likelihood, col = block_assignments, pch = 16, main = "Likelihood-Based Embedding")

#GMM on ASE and likelihood-based embeddings
gmm_ase <- Mclust(X_ase, G = k)
clustering_ase <- gmm_ase$classification

gmm_likelihood <- Mclust(X_likelihood, G = k)
clustering_likelihood <- gmm_likelihood$classification

#ARI
ase_ari <- adjustedRandIndex(block_assignments, clustering_ase)
likelihood_ari <- adjustedRandIndex(block_assignments, clustering_likelihood)

cat("ASE ARI:", ase_ari, "\n")
cat("Likelihood-Based ARI:", likelihood_ari, "\n")
