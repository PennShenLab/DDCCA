import torch
import time


class cca_loss():
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device

    def loss(self, H1, H2, ww, training=True):
        """ reimplement the loss function with testing issue fixed and matrix sqrt bug fixed
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()

        o1 = o2 = H1.size(0)

        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        assert torch.isnan(H1bar).sum().item() == 0
        assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)

        assert torch.isnan(SigmaHat11).sum().item() == 0
        assert torch.isnan(SigmaHat12).sum().item() == 0
        assert torch.isnan(SigmaHat22).sum().item() == 0

        assert len(ww) == 2
        if not training:
            corr = torch.tensor(0.0, device=self.device)
            for i in range(self.outdim_size):  # for each loading component
                w1 = ww[0][:, i]
                w2 = ww[1][:, i]

                crossCov = torch.matmul(torch.matmul(w1.t(), SigmaHat12), w2)
                autoCov = torch.sqrt(torch.matmul(torch.matmul(w1.t(), SigmaHat11), w1)
                                     * torch.matmul(torch.matmul(w2.t(), SigmaHat22), w2))
                corri = crossCov / autoCov
                corr = corr + corri

            return -corr

        D1, V1 = torch.linalg.eigh(SigmaHat11, UPLO='U')  # returns eve an evr of a real symmetric matrix
        D2, V2 = torch.linalg.eigh(SigmaHat22, UPLO='U')

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        # Add small values to increase stability based on the DCCA code discussion
        [UU, DD, VV] = torch.linalg.svd(Tval)
        VV = VV.T
        ww[0] = torch.matmul(SigmaHat11RootInv.detach(), UU[:, 0:self.outdim_size].detach())
        ww[1] = torch.matmul(SigmaHat22RootInv.detach(), VV[:, 0:self.outdim_size].detach())
        corr = DD[0:self.outdim_size].sum()

        return -corr
