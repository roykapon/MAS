import os, torch
from scipy import linalg
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def calculate_precision(generated_features, real_features, k=3):
    assert len(generated_features) == len(real_features), "The number of generated features and real features must be the same."
    return manifold_estimate(real_features, generated_features, k)

def calculate_recall(generated_features, real_features, k=3):
    assert len(generated_features) == len(real_features), "The number of generated features and real features must be the same."
    return manifold_estimate(generated_features, real_features, k)


def manifold_estimate(A_features, B_features, k):
    A_features = list(A_features)
    B_features = list(B_features)
    KNN_list_in_A = {}
    for A_i, A in tqdm(enumerate(A_features), ncols=80):
        pairwise_distances = np.zeros(shape=(len(A_features)))

        for i, A_prime in enumerate(A_features):
            d = linalg.norm((A - A_prime), 2)
            pairwise_distances[i] = d

        v = np.partition(pairwise_distances, k)[k]
        KNN_list_in_A[A_i] = v

    n = 0

    for B in tqdm(B_features, ncols=80):
        for A_prime_i, A_prime in enumerate(A_features):
            d = linalg.norm((B - A_prime), 2)
            if d <= KNN_list_in_A[A_prime_i]:
                n += 1
                break

    return n / len(B_features)


def calculate_gaussian_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; " "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_frechet_distance(batch1, batch2):
    mu1 = np.mean(batch1, axis=0)
    sigma1 = np.cov(batch1, rowvar=False)
    mu2 = np.mean(batch2, axis=0)
    sigma2 = np.cov(batch2, rowvar=False)
    return calculate_gaussian_frechet_distance(mu1, sigma1, mu2, sigma2)


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()
