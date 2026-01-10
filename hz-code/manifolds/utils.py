from typing import Tuple, Any
import torch


eps = 1e-8


def sqrt(x: torch.Tensor) -> torch.Tensor:
    x = clamp(x, min=1e-9)
    return torch.sqrt(x)


class LeakyClamp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, min: float, max: float) -> torch.Tensor:
        with torch.no_grad():
            ctx.save_for_backward(x.ge(min) & x.le(max))
            return torch.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        return grad_output * mask + grad_output * (1 - mask) * eps, None, None


def clamp(x: torch.Tensor, min: float = float("-inf"), max: float = float("+inf")) -> torch.Tensor:
    return LeakyClamp.apply(x, min, max)


class Acosh(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = clamp(x, min=1 + eps)
            z = sqrt(x * x - 1.)
            ctx.save_for_backward(z)
            return torch.log(x + z)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        z, = ctx.saved_tensors
        z_ = z
        return grad_output / z_


def acosh(x: torch.Tensor) -> torch.Tensor:
    return Acosh.apply(x)
