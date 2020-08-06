# the code is from
# https://github.com/allenai/allennlp/blob/c09833c3a2b2fe66f10ffd18761f90d0912c5ea2/allennlp/modules/scalar_mix.py#L10
from typing import List
import torch
from torch.nn import ParameterList, Parameter
from bert_module.scalar_util import tiny_value_of_dtype

class ScalarMix(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors, `mixture = gamma * sum(s_k * tensor_k)`
    where `s = softmax(w)`, with `w` and `gamma` scalar parameters.
    In addition, if `do_layer_norm=True` then apply layer normalization to each tensor
    before weighting.
    """

    def __init__(
        self,
        mixture_size: int,
        do_layer_norm: bool = False,
        initial_scalar_parameters: List[float] = None,
        trainable: bool = True,
        ) -> None:

        super().__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        else:
            assert (len(initial_scalar_parameters) == self.mixture_size)

        self.scalar_parameters = ParameterList(
            [
                Parameter(
                    torch.FloatTensor([initial_scalar_parameters[i]]), requires_grad=trainable
                )
                for i in range(mixture_size)
            ]
        )

        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

    def forward(self, tensors: List[torch.Tensor], mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        Compute a weighted average of the `tensors`.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.
        When `do_layer_norm=True`, the `mask` is required input.  If the `tensors` are
        dimensioned  `(dim_0, ..., dim_{n-1}, dim_n)`, then the `mask` is dimensioned
        `(dim_0, ..., dim_{n-1})`, as in the typical case with `tensors` of shape
        `(batch_size, timesteps, dim)` and `mask` of shape `(batch_size, timesteps)`.
        When `do_layer_norm=False` the `mask` is ignored.
        """
        assert (len(tensors) == self.mixture_size)

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = (
                torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2) / num_elements_not_masked
            )
            return (tensor - mean) / torch.sqrt(variance + tiny_value_of_dtype(variance.dtype))

        normed_weights = torch.nn.functional.softmax(
            torch.cat([parameter for parameter in self.scalar_parameters]), dim=0
        )

        if torch.cuda.is_available():
            normed_weights = normed_weights.cuda()
            self.gamma = self.gamma.to('cuda')

        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)

            return self.gamma * sum(pieces)

        else:
            broadcast_mask = mask.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            # pad对应的位置是0， token对应的是1， sum(mask) token 的总个数， input_dim:768
            num_elements_not_masked = torch.sum(mask) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(
                    weight * _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked)
                )

            # print("gamma:",self.gamma)
            # print("gamma:",self.gamma.grad)
            # print("gamma:",self.gamma.requires_grad)
            # print("normed_weights:", (normed_weights[0]))
            # print("normed_weights:", (normed_weights[0]).requires_grad)
            # print("normed_weights:", (normed_weights[0]).grad)

            return self.gamma * sum(pieces)