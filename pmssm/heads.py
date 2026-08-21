"""Acquisition heads: what the surrogate's single output *means*, and how the
loop turns MC samples of it into an acquisition score.

The production loop regresses the transformed target t = log(Y / true_value) and
thresholds at t = 0 when a verdict is wanted.  The literature's active-learning
results on BSM parameter spaces instead train a *classifier* of the discrete
label 1[t > 0] and acquire where the committee is least sure which side a point
falls on.  Both are wanted here, so a head is a small strategy object rather
than a branch: it declares the training target, the loss, how a raw output is
read, and which acquisition scores it can produce.

Adding a head means adding one class and registering it; nothing else in the
pipeline needs to know about it.

Diagnostics parity
------------------
Every head keeps the same network shape (one scalar output) and the same
decision point (raw output > 0), because a logit is positive exactly when
p > 0.5.  That is deliberate: it means the existing per-iteration diagnostics
that threshold the model's transformed-space output keep working on a
classification-head checkpoint with no changes at all, in particular
``_classification_accuracy`` in ``scripts/plot_hit_rate_trajectories_multiseed.py``
and everything built on it (accuracy trajectories, hit/no-hit plots, the
confusion analysis).  Uncertainty diagnostics that only *rank* sigma, such as
the Spearman rho(sigma, |err|) column of the UQ table and the AUSE curve, also
transfer, since ranking is invariant to the units of the score.

What does not transfer is any metric that reads the output as a *value* of t:
MSE and R^2 in transformed or physical space, NLPD, CRPS and the calibration
columns.  Those are meaningless for a logit, so a head declares
``value_metrics`` and callers record NaN rather than a number that would look
plausible and be wrong (NaN, not None, because the pipeline already treats NaN
as "not computed" and carries it through logs, state arrays and plots).  Use :func:`head_for_run` to recover the head of an
existing run directory; every run made before heads existed reports
``regression``, so old analyses are unaffected.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "HEADS",
    "AcquisitionHead",
    "LeastSquaresVerdictHead",
    "RegressionHead",
    "VerdictClassificationHead",
    "get_head",
    "head_for_run",
    "head_names",
]


class AcquisitionHead:
    """Interface shared by every head.

    Subclasses are stateless; instances are cheap and carry only the target
    configuration they need (the decision threshold in transformed space).
    """

    name = "abstract"
    #: True when the raw output is an estimate of t itself, so that MSE, R^2,
    #: NLPD, CRPS and physical-space inversion are meaningful.
    value_metrics = False
    #: Acquisition scores this head can produce, in the order they are preferred.
    scores: tuple[str, ...] = ()

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)

    # -- training ----------------------------------------------------------
    def make_targets(self, t: torch.Tensor) -> torch.Tensor:
        """Training targets from the transformed target t (shape (N, 1))."""
        raise NotImplementedError

    def criterion(self) -> nn.Module:
        raise NotImplementedError

    # -- reading the output ------------------------------------------------
    def verdict_score(self, raw: torch.Tensor) -> torch.Tensor:
        """Monotone score whose sign at ``threshold`` gives the verdict.

        The identity for every head defined here, which is what makes the
        existing threshold-at-zero diagnostics head-agnostic.
        """
        return raw

    def in_band_probability(self, raw_samples: torch.Tensor, half_width: float):
        """P(|t| < half_width) implied by the MC samples, or None if the head
        cannot express it (a verdict classifier knows a side, not a distance)."""
        return None

    # -- acquisition -------------------------------------------------------
    def summarise(self, raw_samples: torch.Tensor) -> dict:
        """Reduce (T, N) MC samples to the quantities acquisition consumes.

        Returns a dict with at least ``mean`` and ``uncertainty`` (both (N,)),
        plus any head-specific extras.  ``uncertainty`` is the quantity a
        top-k rule would rank by, in whatever units the head works in.
        """
        raise NotImplementedError

    def summarise_gaussian(self, mean: torch.Tensor, var: torch.Tensor) -> dict:
        """Reduce a *Gaussian* latent posterior to the same quantities.

        The GP surrogates return a closed-form posterior rather than MC
        samples, so the head needs a second entry point.  A leading dimension
        is treated as a mixture over sample paths, which is what a deep GP
        returns: the predictive is a mixture of Gaussians, not a Gaussian, and
        collapsing it to its moments first would throw away exactly the
        between-path disagreement that BALD is meant to measure.
        """
        raise NotImplementedError

    def acquisition_score(self, summary: dict, which: str) -> torch.Tensor:
        """A named acquisition score, higher meaning more worth labelling."""
        raise NotImplementedError


class RegressionHead(AcquisitionHead):
    """Production head: MSE on t, acquire by predictive variance.

    ``variance`` is the raw MC-dropout variance; the tolerance pre-filter and
    the proximity weighting that production applies on top of it live in
    :mod:`pmssm.selection`, unchanged, because they are properties of the
    selector rather than of the head.
    """

    name = "regression"
    value_metrics = True
    scores = ("variance",)

    def make_targets(self, t):
        return t

    def criterion(self):
        return nn.MSELoss()

    def in_band_probability(self, raw_samples, half_width):
        return (raw_samples.abs() < half_width).float().mean(dim=0)

    def summarise(self, raw_samples):
        mean = raw_samples.mean(dim=0)
        return {"mean": mean, "var": raw_samples.var(dim=0),
                "uncertainty": raw_samples.var(dim=0)}

    def acquisition_score(self, summary, which):
        if which == "variance":
            return summary["var"]
        raise ValueError(f"{self.name} head has no score {which!r}")


class VerdictClassificationHead(AcquisitionHead):
    """Binary cross-entropy on 1[t > threshold], acquire by committee doubt.

    The raw output is a logit, so ``verdict_score`` is again the identity and
    the decision point is again 0.  Two scores are exposed because they answer
    different questions and the difference is itself a measurement:

    ``entropy``  H[p_bar] of the MC-dropout mean probability.  This is the
                 criterion the literature uses ("committee mean nearest 0.5").
                 By the decomposition H[E p] = I(y; theta) + E H[p] it is the
                 *total* predictive entropy, so it peaks on the decision
                 boundary even when the committee agrees perfectly: it is
                 mostly a statement about the mean, not about disagreement.
    ``bald``     I(y; theta) = H[p_bar] - E_t H[p_t], the epistemic part alone,
                 which is what an uncertainty estimator actually contributes.

    Comparing the two separates "the score is anchored at the boundary" from
    "the committee knows where it is ignorant", which is the question our
    ensemble and Laplace substitutions left open.
    """

    name = "classification"
    value_metrics = False
    scores = ("entropy", "bald")

    def __init__(self, threshold: float = 0.0, link: str = "logistic"):
        super().__init__(threshold=threshold)
        if link not in ("logistic", "probit"):
            raise ValueError(f"unknown link {link!r}; expected 'logistic' or 'probit'")
        #: The neural path trains ``BCEWithLogitsLoss``, so its raw output is a
        #: logit and the link is logistic.  GPyTorch's ``BernoulliLikelihood``
        #: is probit, p(y=1|f) = Phi(f), so a GP arm must score with the link
        #: its own likelihood uses or the acquisition would disagree with the
        #: model it came from.  Both are monotone in f, so the verdict and every
        #: threshold-at-zero diagnostic are identical either way.
        self.link = link

    def _link_fn(self):
        return torch.sigmoid if self.link == "logistic" else _standard_normal_cdf

    def make_targets(self, t):
        return (t > self.threshold).float()

    def criterion(self):
        return nn.BCEWithLogitsLoss()

    def summarise(self, raw_samples):
        p = self._link_fn()(raw_samples.double())
        p_bar = p.mean(dim=0)
        total = _bernoulli_entropy(p_bar)
        aleatoric = _bernoulli_entropy(p).mean(dim=0)
        return {
            "mean": raw_samples.mean(dim=0),   # mean logit, for threshold-at-0 use
            "p_mean": p_bar.float(),
            "var": p.var(dim=0).float(),
            "entropy": total.float(),
            "mutual_information": (total - aleatoric).float(),
            "uncertainty": total.float(),
        }

    def summarise_gaussian(self, mean, var):
        return _summarise_link_gaussian(mean, var, self._link_fn())

    def acquisition_score(self, summary, which):
        if which == "entropy":
            return summary["entropy"]
        if which == "bald":
            return summary["mutual_information"]
        raise ValueError(f"{self.name} head has no score {which!r}")


def _bernoulli_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-12, 1.0 - 1e-12)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def _standard_normal_cdf(f: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(f / np.sqrt(2.0)))


def _gauss_hermite(n: int = 32):
    """Nodes and weights for E[g(f)], f ~ N(mu, var), as a plain sum.

    E[g(f)] = sum_i w_i g(mu + sqrt(2) sigma x_i), with the 1/sqrt(pi) folded
    into the weights.  Quadrature rather than sampling because the acquisition
    ranks a million candidates and MC noise on a score is indistinguishable
    from signal in a top-k rule.
    """
    x, w = np.polynomial.hermite.hermgauss(n)
    return torch.as_tensor(x, dtype=torch.float64), \
        torch.as_tensor(w / np.sqrt(np.pi), dtype=torch.float64)


def _summarise_link_gaussian(mean, var, link, n_quad=32):
    """Shared body of :meth:`summarise_gaussian` for a squashing link.

    ``mean``/``var`` are (N,) or (S, N); the leading dimension is a mixture,
    so the probability and the aleatoric entropy are averaged over it after the
    quadrature, never before.
    """
    mu = mean.double()
    sd = var.double().clamp_min(0).sqrt()
    if mu.dim() == 1:
        mu, sd, squeezed = mu.unsqueeze(0), sd.unsqueeze(0), True
    else:
        squeezed = False
    x, w = _gauss_hermite(n_quad)
    x, w = x.to(mu.device), w.to(mu.device)
    # (S, N, Q) grid of link values
    f = mu.unsqueeze(-1) + np.sqrt(2.0) * sd.unsqueeze(-1) * x
    p = link(f)
    p_path = (p * w).sum(-1)                       # E_f[p] per sample path
    h_path = (_bernoulli_entropy(p) * w).sum(-1)   # E_f[H(p)] per sample path
    p_bar = p_path.mean(0)
    aleatoric = h_path.mean(0)
    total = _bernoulli_entropy(p_bar)
    mean_latent = mu.mean(0)
    var_latent = (sd.pow(2) + mu.pow(2)).mean(0) - mean_latent.pow(2)
    out = {
        "mean": mean_latent.float(),
        "p_mean": p_bar.float(),
        "var": var_latent.float(),
        "entropy": total.float(),
        "mutual_information": (total - aleatoric).clamp_min(0).float(),
        "uncertainty": total.float(),
    }
    return {k: (v.squeeze(0) if squeezed and v.dim() > 1 else v) for k, v in out.items()}


class LeastSquaresVerdictHead(AcquisitionHead):
    """Least-squares classification: regress the +-1 verdict, read it as probit.

    A Bernoulli likelihood is non-conjugate, so an exact GP cannot carry one
    without becoming a variational or Laplace approximation, which would change
    the inference scheme at the same time as the head and confound the
    comparison this arm exists to make.  Regressing the label +-1 under the
    Gaussian likelihood keeps exact conjugate inference untouched and changes
    only the training target; the verdict is then read off the latent with a
    probit link, whose scale is fixed by the +-1 targets rather than being a
    free parameter.  See Rasmussen & Williams, chapter 6.

    The cost is a deliberately misspecified likelihood: Bernoulli variance is
    p(1-p), heteroscedastic and bounded, and a single fitted noise splits the
    difference between the pure-label interior and the mixed-label boundary.
    That is the price of holding the inference fixed, and it is why this head is
    reported alongside the Bernoulli deep GP rather than instead of it.
    """

    name = "lsq_classification"
    value_metrics = False
    scores = ("entropy", "bald")

    def make_targets(self, t):
        return torch.where(t > self.threshold, 1.0, -1.0).to(t.dtype)

    def criterion(self):
        return nn.MSELoss()

    def summarise(self, raw_samples):
        return _summarise_link_gaussian(
            raw_samples.mean(dim=0),
            raw_samples.var(dim=0),
            lambda f: _standard_normal_cdf(f),
        )

    def summarise_gaussian(self, mean, var):
        return _summarise_link_gaussian(mean, var, _standard_normal_cdf)

    def acquisition_score(self, summary, which):
        if which == "entropy":
            return summary["entropy"]
        if which == "bald":
            return summary["mutual_information"]
        raise ValueError(f"{self.name} head has no score {which!r}")


HEADS = {
    RegressionHead.name: RegressionHead,
    VerdictClassificationHead.name: VerdictClassificationHead,
    LeastSquaresVerdictHead.name: LeastSquaresVerdictHead,
}


def head_names() -> list[str]:
    return sorted(HEADS)


def get_head(name: str, threshold: float = 0.0, **kwargs) -> AcquisitionHead:
    """Instantiate a head by name.  ``regression`` reproduces production.

    Extra keyword arguments are passed to the head, which is how a GP arm
    selects the probit link its likelihood uses.
    """
    try:
        cls = HEADS[name]
    except KeyError:
        raise ValueError(f"unknown head {name!r}; known: {head_names()}") from None
    return cls(threshold=threshold, **kwargs)


def head_for_run(config: dict | None) -> str:
    """The head a run used, defaulting to ``regression``.

    Every run produced before heads existed has no such config key, and every
    one of them regressed the target, so the default keeps old analyses correct
    without special-casing them.
    """
    if not config:
        return RegressionHead.name
    return str(config.get("head", RegressionHead.name))
