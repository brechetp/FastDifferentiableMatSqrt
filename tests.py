import unittest
from .torch_utils import MPA_Lya_2D, MPA_Lya_Inv_2D, MPA_Lya, MPA_Lya_Inv
import numpy.testing as npt
import torch

sqrtm = MPA_Lya_2D.apply
sqrtm_inv = MPA_Lya_Inv_2D.apply
bsqrtm = MPA_Lya.apply # batch versions
bsqrtm_inv = MPA_Lya_Inv.apply

rtol = 1e-4
atol = 1e-4
def bsqrtm_eigen(As):
    Rs = torch.zeros_like(As)
    N, n, _ = As.size()
    for i in range(N):
        Lambda, Omega = torch.linalg.eigh(As[i, :, :])
        Rs[i, :, : ] = Omega @ (Lambda.view(-1, 1).sqrt() * Omega.T)
    return Rs

def bsqrtm_eigen_inv(As):
    Rs = torch.zeros_like(As)
    N, n, _ = As.size()
    for i in range(N):
        Lambda, Omega = torch.linalg.eigh(As[i, :, :])
        Rs[i, :, : ] = Omega @ ((1/Lambda.view(-1, 1).sqrt()) * Omega.T)
    return Rs


class TestMatSqrt(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = 50
        self.N = 100
        As = torch.randn(self.N, self.n, self.n)
        self.As = As.bmm(As.transpose(2, 1))

    def test_batch_mat_sqrt(self):
        # A = torch_utils.
        Rs = bsqrtm(self.As)
        npt.assert_allclose(self.As, Rs.bmm(Rs), rtol=rtol, atol=atol)

    def test_batch_eig_sqrt(self):


        Rs = bsqrtm_eigen(self.As)
        npt.assert_allclose(self.As, Rs.bmm(Rs), rtol=rtol, atol=atol)

    def test_2d_sqrtm(self):

        As = torch.randn(1, self.n, self.n)
        As = As.bmm(As.transpose(2, 1))
        R1 = bsqrtm(As)
        R2 = sqrtm(As.squeeze(0)).unsqueeze(0)
        npt.assert_allclose(R1, R2, rtol=rtol, atol=atol)

    def test_batch_mat_sqrt_inv(self):
        # A = torch_utils.
        Rs = bsqrtm_inv(self.As)
        npt.assert_allclose(self.As, torch.linalg.inv(Rs.bmm(Rs)), rtol=rtol, atol=atol)

    def test_batch_eig_sqrt_inv(self):


        Rs = bsqrtm_eigen_inv(self.As)
        npt.assert_allclose(self.As, torch.linalg.inv(Rs.bmm(Rs)), rtol=rtol, atol=atol)

    def test_2d_sqrtm_inv(self):

        As = torch.randn(1, self.n, self.n)
        As = As.bmm(As.transpose(2, 1))
        R1 = bsqrtm_inv(As)
        R2 = sqrtm_inv(As.squeeze(0)).unsqueeze(0)
        npt.assert_allclose(R1, R2, rtol=rtol, atol=atol)



if __name__ == "__main__":
    unittest.main()
