import torch
from manifolds.utils import acosh, sqrt


EXP_MAX_NORM = 1.5


def inner(u, v, *, keepdim=False, dim=-1):
    return _inner(u, v, keepdim=keepdim, dim=dim)


def inner_multi(u, v, dim: int = -1):
    u0 = u.narrow(dim, 0, 1)
    v0 = v.narrow(dim, 0, 1)
    return u @ v.transpose(0, 1) - 2 * u0 @ v0.transpose(0, 1)


def _inner(u, v, keepdim: bool = False, dim: int = -1):
    d = u.size(dim) - 1
    uv = u * v
    if keepdim is False:
        return -uv.narrow(dim, 0, 1).squeeze(dim) + uv.narrow(
            dim, 1, d
        ).sum(dim=dim, keepdim=False)
    else:
        return -uv.narrow(dim, 0, 1) + uv.narrow(dim, 1, d).sum(
            dim=dim, keepdim=True
        )


def inner0(v, *, k, keepdim=False, dim=-1):
    return _inner0(v, k=k, keepdim=keepdim, dim=dim)


def _inner0(v, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    res = -v.narrow(dim, 0, 1)
    if keepdim is False:
        res = res.squeeze(dim)
    return res


def dist(x, y, *, k, keepdim=False, dim=-1):
    return _dist(x, y, k=k, keepdim=keepdim, dim=dim)


def _dist(x, y, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner(x, y, dim=dim, keepdim=keepdim)
    return acosh(d / k)


def dist0(x, *, k, keepdim=False, dim=-1):
    return _dist0(x, k=k, keepdim=keepdim, dim=dim)


def _dist0(x, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner0(x, k=k, dim=dim, keepdim=keepdim)
    return acosh(d / k)


def cdist(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor):
    x = x.clone()
    x.narrow(-1, 0, 1).mul_(-1)
    return acosh(-(x @ y.transpose(-1, -2)))


def project(x, *, k, dim=-1, max_norm=None):
    return _project(x, k=k, dim=dim, max_norm=max_norm if max_norm is not None else 0)


@torch.jit.script
def _project(x, k: torch.Tensor, dim: int = -1, max_norm: float = 1.5):
    dn = x.size(dim) - 1
    right_ = x.narrow(dim, 1, dn)
    if max_norm:
        right_ = torch.renorm(right_, 2, -2, max_norm)
    left_ = torch.sqrt(
        k + (right_ * right_).sum(dim=dim, keepdim=True)
    )
    x = torch.cat((left_, right_), dim=dim)
    return x


def project_polar(x, *, k, dim=-1):
    return _project_polar(x, k=k, dim=dim)


def _project_polar(x, k: torch.Tensor, dim: int = -1):
    dn = x.size(dim) - 1
    d = x.narrow(dim, 0, dn)
    r = x.narrow(dim, -1, 1)
    res = torch.cat(
        (
            torch.cosh(r / torch.sqrt(k)),
            torch.sqrt(k) * torch.sinh(r / torch.sqrt(k)) * d,
        ),
        dim=dim,
    )
    return res


def project_u(x, v, *, k, dim=-1):
    return _project_u(x, v, k=k, dim=dim)


def _project_u(x, v, k: torch.Tensor, dim: int = -1):
    return v.addcmul(_inner(x, v, dim=dim, keepdim=True), x / k)


def project_u0(u):
    narrowed = u.narrow(-1, 0, 1)
    vals = torch.zeros_like(u)
    vals[..., 0:1] = narrowed
    return u - vals


def norm(u, *, keepdim=False, dim=-1):
    return _norm(u, keepdim=keepdim, dim=dim)


def _norm(u, keepdim: bool = False, dim: int = -1):
    return sqrt(_inner(u, u, keepdim=keepdim))


def expmap(x, u, *, k, dim=-1):
    return _expmap(x, u, k=k, dim=dim)


def _expmap(x, u, k: torch.Tensor, dim: int = -1):
    nomin = (_norm(u, keepdim=True, dim=dim))
    u = u / nomin
    nomin = nomin.clamp_max(EXP_MAX_NORM)
    p = torch.cosh(nomin) * x + torch.sinh(nomin) * u
    return p


def expmap0(u, *, k, dim=-1):
    return _expmap0(u, k, dim=dim)


def _expmap0(u, k: torch.Tensor, dim: int = -1):
    nomin = (_norm(u, keepdim=True, dim=dim))
    u = u / nomin
    nomin = nomin.clamp_max(EXP_MAX_NORM)
    l_v = torch.cosh(nomin)
    r_v = torch.sinh(nomin) * u
    dn = r_v.size(dim) - 1
    p = torch.cat((l_v + r_v.narrow(dim, 0, 1), r_v.narrow(dim, 1, dn)), dim)
    return p


def logmap(x, y, *, k, dim=-1):
    return _logmap(x, y, k=k, dim=dim)


def _logmap(x, y, k, dim: int = -1):
    dist_ = _dist(x, y, k=k, dim=dim, keepdim=True)
    nomin = y + 1.0 / k * _inner(x, y, keepdim=True) * x
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom


def logmap0(y, *, k, dim=-1):
    return _logmap0(y, k=k, dim=dim)


def _logmap0(y, k, dim: int = -1):
    alpha = -_inner0(y, k, keepdim=True)
    zero_point = torch.zeros(y.shape[-1], device=y.device)
    zero_point[0] = 1
    return acosh(alpha) / torch.sqrt(alpha * alpha - 1) * (y - alpha * zero_point)


def logmap0back(x, *, k, dim=-1):
    return _logmap0back(x, k=k, dim=dim)


def _logmap0back(x, k, dim: int = -1):
    dist_ = _dist0(x, k=k, dim=dim, keepdim=True)
    nomin_ = 1.0 / k * _inner0(x, k=k, keepdim=True) * x
    dn = nomin_.size(dim) - 1
    nomin = torch.cat(
        (nomin_.narrow(dim, 0, 1) + 1, nomin_.narrow(dim, 1, dn)), dim
    )
    denom = _norm(nomin, keepdim=True)
    return dist_ * nomin / denom

def egrad2rgrad(x, grad, *, k, dim=-1):
    return _egrad2rgrad(x, grad, k=k, dim=dim)


def _egrad2rgrad(x, grad, k, dim: int = -1):
    grad.narrow(-1, 0, 1).mul_(-1)
    grad = grad.addcmul(_inner(x, grad, dim=dim, keepdim=True), x / k)
    return grad


def parallel_transport(x, y, v, *, k, dim=-1):
    return _parallel_transport(x, y, v, k=k, dim=dim)


def _parallel_transport(x, y, v, k, dim: int = -1):
    nom = _inner(y, v, keepdim=True)
    denom = torch.clamp_min(k - _inner(x, y, keepdim=True), 1e-7)
    return v.addcmul(nom / denom, x + y)


def parallel_transport0(y, v, *, k, dim=-1):
    return _parallel_transport0(y, v, k=k, dim=dim)


def _parallel_transport0(y, v, k, dim: int = -1):
    nom = _inner(y, v, keepdim=True)
    denom = torch.clamp_min(k - _inner0(y, k=k, keepdim=True), 1e-7)
    zero_point = torch.zeros_like(y)
    zero_point[..., 0] = 1
    return v.addcmul(nom / denom, y + zero_point)


def parallel_transport0back(x, v, *, k, dim: int = -1):
    return _parallel_transport0back(x, v, k=k, dim=dim)


def _parallel_transport0back(x, v, k, dim: int = -1):
    nom = _inner0(v, k=k, keepdim=True)
    denom = torch.clamp_min(k - _inner0(x, k=k, keepdim=True), 1e-7)
    zero_point = torch.zeros_like(x)
    zero_point[..., 0] = 1
    return v.addcmul(nom / denom, x + zero_point)


def geodesic_unit(t, x, u, *, k):
    return _geodesic_unit(t, x, u, k=k)


def _geodesic_unit(t, x, u, k):
    return (
        torch.cosh(t) * x
        + torch.sinh(t) * u
    )


def lorentz_to_poincare(x, k, dim=-1):
    dn = x.size(dim) - 1
    return x.narrow(dim, 1, dn) / (x.narrow(dim, 0, 1) + 1)


def poincare_to_lorentz(x, k, dim=-1, eps=1e-6):
    x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
    res = (
        torch.cat((1 + x_norm_square, 2 * x), dim=dim)
        / (1.0 - x_norm_square + eps)
    )
    return res