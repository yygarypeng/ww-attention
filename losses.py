import torch
import torch.nn.functional as F

# RBF kernel widths
SIGMA_LST = [0.05, 0.1, 0.5, 1.0, 5.0]


def compute_mmd(x, y, bandwidth_range=SIGMA_LST):
    """
    Maximum Mean Discrepancy (Gaussian kernel) between tensors x and y.
    x, y: (...,) or (N,) shaped tensors.
    """
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    
    with torch.no_grad(): # a heuristic way to set bandwidths
        dists = torch.cdist(y, y, p=2)
        median_dist = torch.median(dists)
        # print("Median distance: ", median_dist.item()) # debug
        # bandwidth_range = [0.03*median_dist, 0.07*median_dist, 0.3*median_dist, 0.7*median_dist, 3*median_dist, 7*median_dist]
        bandwidth_range = [s * median_dist for s in bandwidth_range]
    
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * xy
    
    XX = torch.zeros_like(xx)
    YY = torch.zeros_like(yy)
    XY = torch.zeros_like(xy)

    for a in bandwidth_range:
        XX += torch.exp(-0.5 * dxx / a**2)
        YY += torch.exp(-0.5 * dyy / a**2)
        XY += torch.exp(-0.5 * dxy / a**2)
        
    #     print("bandwidth: ", a)
    #     print("Tot:", torch.mean(XX + YY - 2. * XY))
    # raise Exception("DEBUG: stop here")
    
    return torch.mean(XX + YY - 2. * XY)


def invariant_mass(fourvec):
    """
    fourvec: (..., 4) where order is (px, py, pz, E)
    returns: (...,) positive mass = sqrt(|E^2 - |p|^2|)
    """
    px, py, pz, E = fourvec[..., 0], fourvec[..., 1], fourvec[..., 2], fourvec[..., 3]
    mass2 = E * E - (px * px + py * py + pz * pz)
    return torch.sqrt(mass2.abs().clamp(min=1e-16))


def _alpha_from_p4(lep_on_p4, neu_on_p4, lep_off_p4, neu_off_p4):
    def _cal_norm(p4):
        return torch.sqrt((p4[..., :3] ** 2).sum(dim=-1).clamp(min=1e-16))

    dinu_lep_on_p4 = lep_on_p4 + neu_on_p4 + neu_off_p4
    dinu_lep_off_p4 = lep_off_p4 + neu_on_p4 + neu_off_p4
    dinu_lep_on_mass = invariant_mass(dinu_lep_on_p4)
    dinu_lep_off_mass = invariant_mass(dinu_lep_off_p4)

    denom = _cal_norm(neu_on_p4) + _cal_norm(neu_off_p4) + 1e-16
    alpha_on = _cal_norm(neu_on_p4) / denom
    alpha_off = _cal_norm(neu_off_p4) / denom
    return torch.where(dinu_lep_on_mass > dinu_lep_off_mass, alpha_on, alpha_off)


def alpha_mmd_loss(x_batch, y_true, y_pred, w_mass=80.379):
    l0, l1 = x_batch[..., :4], x_batch[..., 4:8]
    w0_true, w1_true = y_true[..., :4], y_true[..., 4:8]
    w0_pred, w1_pred = y_pred[..., :4], y_pred[..., 4:8]

    w_mass_con2 = w_mass ** 2
    w0_true_mass2 = invariant_mass(w0_true) ** 2
    w1_true_mass2 = invariant_mass(w1_true) ** 2
    true_0_on_mask = (w0_true_mass2 - w_mass_con2).abs() < (w1_true_mass2 - w_mass_con2).abs()

    w0_pred_mass2 = invariant_mass(w0_pred) ** 2
    w1_pred_mass2 = invariant_mass(w1_pred) ** 2
    pred_0_on_mask = (w0_pred_mass2 - w_mass_con2).abs() < (w1_pred_mass2 - w_mass_con2).abs()

    def _build_neu_p4(w_p4, lep_p4):
        nu_3 = w_p4[..., :3] - lep_p4[..., :3]
        zeros = torch.zeros_like(lep_p4[..., :1])
        return torch.cat([nu_3, zeros], dim=-1)

    true_neu0 = _build_neu_p4(w0_true, l0)
    true_neu1 = _build_neu_p4(w1_true, l1)
    pred_neu0 = _build_neu_p4(w0_pred, l0)
    pred_neu1 = _build_neu_p4(w1_pred, l1)

    true_lep_on = torch.where(true_0_on_mask.unsqueeze(-1), l0, l1)
    true_lep_off = torch.where(true_0_on_mask.unsqueeze(-1), l1, l0)
    true_neu_on = torch.where(true_0_on_mask.unsqueeze(-1), true_neu0, true_neu1)
    true_neu_off = torch.where(true_0_on_mask.unsqueeze(-1), true_neu1, true_neu0)

    pred_neu_on = torch.where(pred_0_on_mask.unsqueeze(-1), pred_neu0, pred_neu1)
    pred_neu_off = torch.where(pred_0_on_mask.unsqueeze(-1), pred_neu1, pred_neu0)

    alpha_true = _alpha_from_p4(true_lep_on, true_neu_on, true_lep_off, true_neu_off)
    alpha_pred = _alpha_from_p4(true_lep_on, pred_neu_on, true_lep_off, pred_neu_off)

    return compute_mmd(alpha_pred, alpha_true)


def mae_loss(y_true, y_pred):
    # do not consider mass targets in y_true
    return F.l1_loss(y_pred[..., :8], y_true[..., :8])

def huber_loss(y_true, y_pred):
    # do not consider mass targets in y_true
    return F.huber_loss(y_pred[..., :8], y_true[..., :8])


def uncertainty_huber_nll_loss(y_true, y_pred, log_b_pred, delta=1.0, log_b_min=-6.0, log_b_max=3.0):
    y_t = y_true[..., :8]
    y_p = y_pred[..., :8]
    log_b = torch.clamp(log_b_pred[..., :8], min=log_b_min, max=log_b_max)

    pointwise_huber = F.huber_loss(y_p, y_t, reduction="none", delta=delta)
    weighted = torch.exp(-log_b) * pointwise_huber + log_b
    return weighted.mean()

def neg_r2_loss(y_true, y_pred):
    y_t = y_true[..., :8]
    y_p = y_pred[..., :8]
    ss_res = torch.sum((y_t - y_p) ** 2)
    ss_tot = torch.sum((y_t - torch.mean(y_t)) ** 2)
    return ss_res / ss_tot - 1.0


def w_mass_mae_losses(y_true, y_pred):
    """
    Returns: (w0_mae, w1_mae) using true W masses at y_true[..., 8], y_true[..., 9]
    """
    w0_pred, w1_pred = y_pred[..., :4], y_pred[..., 4:8]
    w0_true_mass, w1_true_mass = y_true[..., 8], y_true[..., 9]

    w0_mass = invariant_mass(w0_pred)
    w1_mass = invariant_mass(w1_pred)

    return (
        torch.mean(torch.abs(w0_mass - w0_true_mass)),
        torch.mean(torch.abs(w1_mass - w1_true_mass)),
    )


def w_mass_mmd_losses(y_true, y_pred):
    """
    Returns: (mmd_w0, mmd_w1) comparing predicted mass distributions to truth.
    """
    w0_pred, w1_pred = y_pred[..., :4], y_pred[..., 4:8]
    w0_true_mass, w1_true_mass = y_true[..., 8], y_true[..., 9]

    w0_mass = invariant_mass(w0_pred)
    w1_mass = invariant_mass(w1_pred)

    return compute_mmd(w0_mass, w0_true_mass), compute_mmd(w1_mass, w1_true_mass)


def higgs_mass_loss(y_pred):
    """
    Higgs mass from sum of two W four-vectors; target 125.0 (GeV)
    """
    w0, w1 = y_pred[..., :4], y_pred[..., 4:8]
    higgs_4 = w0 + w1
    h_mass = invariant_mass(higgs_4)
    return torch.clamp(torch.mean(h_mass - 125.0).abs(), min=1e-16)


def nu_mass_loss(x_batch, y_pred):
    """
    n0_4vect = y_pred[..., :4] - x_batch[..., :4]
    n1_4vect = y_pred[..., 4:8] - x_batch[..., 4:8]
    penalize neutrino invariant mass (prefer ~0)
    """
    n0_4 = y_pred[..., :4] - x_batch[..., :4]
    n1_4 = y_pred[..., 4:8] - x_batch[..., 4:8]

    nu0_mass = invariant_mass(n0_4)
    nu1_mass = invariant_mass(n1_4)
    return torch.mean(nu0_mass + nu1_mass)


def aux_mom_mmd_loss(y_true, y_pred, epoch):
    """
    Returns: (mmd_w0, mmd_w1) comparing predicted momentum distributions to truth.
    """
    w0_pred, w1_pred = y_pred[..., :4], y_pred[..., 4:8]
    w0_true, w1_true = y_true[..., :4], y_true[..., 4:8]
    w = min(epoch / 30.0, 1.0) # linearly increase weight over epochs

    return w*compute_mmd(w0_pred, w0_true, [0.1, 0.5, 1.0, 5.0]), w*compute_mmd(w1_pred, w1_true, [0.1, 0.5, 1.0, 5.0])

def dinu_pt_loss(x_batch, y_pred):
    """
    Di-neutrino pT consistency with MET in inputs:
    x[..., 8] = MET px, x[..., 9] = MET py  (as in your TF code)
    """
    n0_4 = y_pred[..., :4] - x_batch[..., :4]
    n1_4 = y_pred[..., 4:8] - x_batch[..., 4:8]
    nn_4 = n0_4 + n1_4
    nn_px_diff = torch.clamp((nn_4[..., 0] - x_batch[..., 8]).abs(), min=1e-10)
    nn_py_diff = torch.clamp((nn_4[..., 1] - x_batch[..., 9]).abs(), min=1e-10)
    return torch.mean(nn_px_diff + nn_py_diff)

def gaussian_log_prob(y, mean, log_std):
    var = torch.exp(2.0 * log_std)
    return -0.5 * ((y - mean) ** 2 / var + 2.0 * log_std + torch.log(2 * torch.pi))

def laplace_log_prob(y, mean, log_b):
    b = torch.exp(log_b)
    return -torch.abs(y - mean) / b - log_b - torch.log(2.0)

def gaussian_laplace_mixture_nll(
    y_true,
    mean1, log_std1,
    mean2, log_b2,
    logit_pi
):
    """
    Gaussian + Laplace mixture NLL
    
    Args:
        y_true: (B, D)
        mean1, log_std1: Gaussian parameters
        mean2, log_b2: Laplace parameters
        logit_pi: (B, 1) or (B, D) mixture logits
    
    Returns:
        scalar NLL
    """

    # log probabilities per dimension
    log_p_gauss = gaussian_log_prob(y_true[:, :8], mean1, log_std1)
    log_p_laplace = laplace_log_prob(y_true[:, :8], mean2, log_b2)

    # sum over dimensions
    log_p_gauss = log_p_gauss.sum(dim=-1)
    log_p_laplace = log_p_laplace.sum(dim=-1)

    # mixture weights
    log_pi = F.logsigmoid(logit_pi.squeeze(-1))
    log_1m_pi = F.logsigmoid(-logit_pi.squeeze(-1))

    # log-sum-exp for numerical stability
    log_mix = torch.logsumexp(
        torch.stack([
            log_pi + log_p_gauss,
            log_1m_pi + log_p_laplace
        ], dim=0),
        dim=0
    )

    return -log_mix.mean()

    
    
