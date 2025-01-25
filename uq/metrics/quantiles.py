import torch

from uq import utils

log = utils.get_logger(__name__)


def quantile_scores(y, quantiles, alpha):
    """
    Return the quantile score for a list of quantile levels.
    Taking the mean of this score over the levels will give a proper scoring rule for each levels.
    """
    batch_size, n_levels = quantiles.shape
    if alpha.dim() == 1:   # If alpha is the same for each elements of the batch
        alpha = alpha[None, :]   # We add the batch dimension
    assert alpha.shape[-1] == n_levels and y.shape == (batch_size,)
    diff = y[:, None] - quantiles
    indicator = (diff < 0).float()
    score_per_quantile = diff * (alpha - indicator)
    return score_per_quantile


def interval_score(y, quantiles, alpha):
    """
    It is supposed that quantile_scores[i] and quantile_scores[-i] have corresponding levels alpha/2 and 1-alpha/2
    """

    batch_size, n_levels = quantiles.shape
    if alpha.dim() == 1:   # If alpha is the same for each elements of the batch
        alpha = alpha[None, :]   # We add the batch dimension
    assert alpha.shape[-1] == n_levels and y.shape == (batch_size,)
    mid = int(alpha.shape[-1] / 2)
    
    interval_score = torch.zeros(batch_size, device=y.device)
    for i in range(mid):
        p = (0.5 - alpha[: , i]) * 2
        pred_l, pred_u = quantiles[:, i], quantiles[:, -i-1]
    
        below_l = ((pred_l - y) > 0).float()
        above_u = ((y - pred_u) > 0).float()
        score_per_p = (
            (pred_u - pred_l)
            + (2.0 / (1 - p)) * (pred_l - y) * below_l
            + (2.0 / (1 - p)) * (y - pred_u) * above_u
        )
        interval_score = score_per_p + interval_score
    
    return interval_score.mean() / mid

def variance(y, quantiles, alpha):

    batch_size, n_levels = quantiles.shape
    assert alpha.shape[-1] == n_levels and y.shape == (batch_size,)
    dp = alpha[1:] - alpha[:-1]
    x = 0.5 * (quantiles.T[1: ] + quantiles.T[:-1])
    expected_values = torch.sum(x.T * dp, 1)
    variance_values = torch.sum(((x - expected_values) ** 2).T * dp, 1)
    return variance_values


def check_score(y, quantiles, alpha):
    """
    Return the check score for a list of quantile levels.
    """
    batch_size, n_levels = quantiles.shape
    if alpha.dim() == 1:   # If alpha is the same for each elements of the batch
        alpha = alpha[None, :]   # We add the batch dimension
    assert alpha.shape[-1] == n_levels and y.shape == (batch_size,)
    diff = quantiles - y[:, None] 
    mask = (diff >= 0).float() - alpha
    check_score_per_quantile = torch.mean(diff * mask, dim=0)
    return check_score_per_quantile.mean()
    


def wis(quantile_scores):
    """
    Args:
        quantile_scores: tensor that contains the quantile scores.
        It is supposed that the corresponding quantile levels are [l_1, l_2, ..., l_k]
        and that l_1 - 0 == l_2 - l_1 == ... == 1 - l_k.
    Returns:
        The weighted interval score, an approximation of the CRPS.
    """
    side_values = (quantile_scores[:, :1] + quantile_scores[:, -1:]) * 3 / 2
    quantile_scores = torch.cat((quantile_scores, side_values), dim=1)
    return 2 * quantile_scores.mean(dim=1)


def crps_helper(mean, std):
    dist = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
    return mean * (2 * dist.cdf(mean / std) - 1) + 2 * std * torch.exp(dist.log_prob(mean / std))


def crps_normal_mixture_from_params(mean, std, w, y):
    batch_size, mixture_size = mean.shape
    term1 = w * crps_helper(y[:, None] - mean, std)
    var = std**2
    factor = crps_helper(
        mean[:, :, None] - mean[:, None, :],
        torch.sqrt(
            var[:, :, None] + var[:, None, :] + 1e-12
        ),  # The derivative of torch.sqrt is infinity in 0
    )
    term2 = w[:, :, None] * w[:, None, :] * factor
    return term1.sum(dim=1) - term2.sum(dim=(1, 2)) / 2


def crps_normal_mixture(dist, y):
    return crps_normal_mixture_from_params(
        dist.component_distribution.loc,
        dist.component_distribution.scale,
        dist.mixture_distribution.probs,
        y,
    )


def length_and_coverage_from_quantiles(y, quantiles, alpha, left_alpha, right_alpha):
    left_alpha_index = (torch.isclose(alpha, torch.tensor(left_alpha))).nonzero().item()
    right_alpha_index = (torch.isclose(alpha, torch.tensor(right_alpha))).nonzero().item()
    left_bound = quantiles[..., left_alpha_index]
    right_bound = quantiles[..., right_alpha_index]
    length = torch.maximum(right_bound - left_bound, torch.tensor(0))
    coverage = ((left_bound < y) & (y < right_bound)).float()
    return length, coverage


def quantile_sharpness_reward(quantiles, alpha):
    """
    The quantile sharpness reward corresponds to the
    """
    mid = quantiles.shape[1] // 2
    assert alpha[:mid] == 1 - alpha[mid + 1 :: -1]
    miscoverage = 2 * alpha[:mid]
    left_bound = quantiles[..., :mid]
    right_bound = quantiles[..., mid:].flip(dims=(-1,))
    length = torch.maximum(right_bound - left_bound, torch.tensor(0))
    weighted_length = miscoverage / 2.0 * length
    return weighted_length.mean(dim=-1)
