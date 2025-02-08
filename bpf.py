from resampling import multinomial_resampling, stratified_resampling, systematic_resampling
import numpy as np
from scipy.special import logsumexp


def run_bpf(y, N, model, resampling_scheme='multinomial', adaptive=False, d=1):
    # Bootstrap Particle Filter (BPF)
    T = len(y)
    d_y = 1

    particles = np.zeros((N, d, T))
    normalized_weights = np.zeros((N, T))
    B = np.zeros((N, T))
    ESS = np.zeros(T)
    log_weights = np.zeros((N, T))
    marg_log_likelihood = 0

    if resampling_scheme.lower() in 'multinomial':
        resampling = multinomial_resampling
    elif resampling_scheme.lower() in 'stratified':
        resampling = stratified_resampling
    elif resampling_scheme.lower() in 'systematic':
        resampling = systematic_resampling
    else:
        assert False, "Unknown resampling scheme"

    particles[..., 0] = model.particle_0(N)
    log_g_t = model.log_g(x=particles[:, :d_y, 0], y=y[0])
    normalized_weights[:, 0], log_weights[:, 0] = update_weights(log_weights=log_weights[:, 0], log_g_t=log_g_t)
    new_ancestors = list(range(N))
    B[:, 0] = new_ancestors

    # == Stats == #
    marg_log_likelihood += logsumexp(log_weights[:, 0] - np.log(N))

    for t in range(1, T):
        # == Resampling == #
        ESS[t - 1] = 1 / np.sum(normalized_weights[:, t - 1] ** 2)
        if resample_criterion(adaptive, ESS[t - 1], N):
            new_ancestors = resampling(normalized_weights[:, t - 1]).astype(int)
            normalized_weights[:, t - 1] = 1 / N
            log_weights[:, t - 1] = 0
        else:
            new_ancestors = list(range(N))

        # == Propagate == #
        B[:, t] = new_ancestors
        particles[:, :, t] = model.propagate(particles[new_ancestors, :, t - 1])

        # == Compute weights == #
        log_g_t = model.log_g(particles[:, :d_y, t], y[t])  # incremental weight function
        normalized_weights[:, t], log_weights[:, t] = update_weights(log_weights[:, t - 1], log_g_t)

        # == Marg. Log-Likelihood == #
        marg_log_likelihood += logsumexp(log_weights[:, t] - np.log(N))

    ESS[-1] = 1 / np.sum(normalized_weights[:, T - 1] ** 2)

    # == Sample sequence from resulting posterior == #
    B = B.astype(int)
    b = np.where(np.random.uniform(size=1) < np.cumsum(normalized_weights[:, T - 1]))[0][0]
    x_star = np.zeros((d, T))
    indx = b
    for t in reversed(range(T)):
        x_star[:, t] = particles[indx, :, t]  # sampled particle trajectory
        indx = B[indx, t]

    out = {'x_star': x_star, 'posterior': normalized_weights[:, -1], 'ESS': ESS,
           'particles': particles, 'B': B, 'marg_log_likelihood': marg_log_likelihood}
    return out


def update_weights(log_weights, log_g_t):
    log_weights += log_g_t
    log_w_tilde = log_weights - logsumexp(log_weights)
    normalized_weights = np.exp(log_w_tilde)
    return normalized_weights, log_weights


def resample_criterion(adaptive, ESS, N):
    if adaptive:
        return ESS < N / 2
    else:
        return True


