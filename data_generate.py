import torch
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import os

torch.manual_seed(1)
np.random.seed(1)

dimX = 10
dimS = dimU = 5
dimS1 = 5
dimS2 = 5
dimS3 = 5
q = 2

# nobs = 2000
# nexp = 2000
# nsim = 200
nobs = 2000
nexp = 2000
nsim = 10

def g(lindat):
    #(-1,0,1)**q
    return torch.sign(lindat) * torch.abs(lindat) ** q

def rnd_norm_coef(dim1, dim2, scale, min=0, max=1):
    coef = torch.zeros(dim1, dim2)
    for j in range(dim2):
        # [min, max)
        vec = torch.rand(dim1) * (max - min) + min
        vec = scale * vec / torch.sqrt(torch.sum(vec ** 2))
        coef[:, j] = vec
    return coef

def rnd_data(n, p):
    data = torch.from_numpy(multivariate_normal.rvs(mean=np.zeros(p), cov=np.eye(p), size=n)).float()
    if p == 1:
        # 返回二维张量 (n, 1)
        data = data.unsqueeze(1)
    return data

res_list = []

if not os.path.exists("tmp3"):
    os.makedirs("tmp3")


for i in range(1, nsim + 1):
    print(i)

    kappaU = rnd_norm_coef(dimU, 1, np.sqrt(0.5))
    kappaX = rnd_norm_coef(dimX, 1, np.sqrt(0.5))
    # 1
    tau1 = rnd_norm_coef(1, dimS, np.sqrt(0.5))
    beta1 = rnd_norm_coef(dimX, dimS, np.sqrt(0.5))
    gamma1 = rnd_norm_coef(dimU, dimS, np.sqrt(0.5))
    # 2
    tau2 = np.sqrt(0.5)
    alpha2 = rnd_norm_coef(dimS, 1, np.sqrt(0.5))
    beta2 = rnd_norm_coef(dimX, 1, np.sqrt(0.5))
    gamma2 = rnd_norm_coef(dimU, 1, np.sqrt(0.5))
    # 3
    tau3 = rnd_norm_coef(1, dimS, np.sqrt(0.5))
    # alpha3 = rnd_norm_coef(1, dimS, np.sqrt(0.5))
    alpha3 = rnd_norm_coef(dimS, dimS, np.sqrt(0.5))
    beta3 = rnd_norm_coef(dimX, dimS, np.sqrt(0.5))
    gamma3 = rnd_norm_coef(dimU, dimS, np.sqrt(0.5))
    # y
    tauy = np.sqrt(0.5)
    alphay = rnd_norm_coef(dimS, 1, np.sqrt(0.5))
    betay = rnd_norm_coef(dimX, 1, np.sqrt(0.5))
    gammay = rnd_norm_coef(dimU, 1, np.sqrt(0.5))

    # print(f"alphay shape: {alphay.shape}")
    # print(f"betay shape: {betay.shape}")
    # print(f"gammay shape: {gammay.shape}")

    # 观测数据
    X = rnd_data(nobs, dimX)
    U = rnd_data(nobs, dimU)
    prob = 1 / (1 + torch.exp(X @ kappaX + U @ kappaU))
    A = (torch.rand(nobs) <= prob.squeeze()).float().unsqueeze(1)
    S1 = A @ tau1 + X @ beta1 + U @ gamma1 + np.sqrt(0.5) * rnd_data(nobs, dimS)
    # S2 = A * tau2 + S1 @ alpha2 + X @ beta2 + U @ gamma2 + np.sqrt(0.5) * rnd_data(nobs, 1)
    S2 = A * tau2 + S1 @ alpha2 + X @ beta2 + U @ gamma2 + np.sqrt(0.5) * rnd_data(nobs, dimS)
    # print("S2 shape:", S2.shape)  # (nobs, dimS) (2000,5)
    S3 = A @ tau3 + S2 @ alpha3 + X @ beta3 + U @ gamma3 + np.sqrt(0.5) * rnd_data(nobs, dimS)
    Y = A * tauy + S3 @ alphay + X @ betay + U @ gammay + np.sqrt(0.5) * rnd_data(nobs, 1)
    noise = np.sqrt(0.5) * rnd_data(nobs, 1)

    # print(f"S3 shape: {S3.shape}")
    # print(f"X shape: {X.shape}")
    # print(f"U shape: {U.shape}")
    # print(f"noise shape {noise.shape}")
    #
    # print(f"g(X) shape: {g(X).shape}")
    # print(f"g(S2) shape: {g(S2).shape}")
    # print(f"g(S1) shape: {g(S1).shape}")
    # print(f"g(S3) shape: {g(S3).shape}")
    # print(f"Y shape: {Y.shape}")
    # print(f"A shape: {A.shape}")
    data_obs = torch.cat((g(X), g(S2), g(S1), g(S3), Y, A), dim=1).numpy()
    pd.DataFrame(data_obs).to_csv(f"tmp3/obs_{i}.csv", index=False, header=False)

    # 实验数据
    X = rnd_data(nexp, dimX)
    U = rnd_data(nexp, dimU)
    A = (torch.rand(nexp) <= 0.5).float().unsqueeze(1)
    S1 = A @ tau1 + X @ beta1 + U @ gamma1 + np.sqrt(0.5) * rnd_data(nexp, dimS)
    # S2 = A * tau2 + S1 @ alpha2 + X @ beta2 + U @ gamma2 + np.sqrt(0.5) * rnd_data(nexp, 1)
    S2 = A * tau2 + S1 @ alpha2 + X @ beta2 + U @ gamma2 + np.sqrt(0.5) * rnd_data(nexp, dimS)
    S3 = A @ tau3 + S2 @ alpha3 + X @ beta3 + U @ gamma3 + np.sqrt(0.5) * rnd_data(nexp, dimS)
    Y = A * tauy + S3 @ alphay + X @ betay + U @ gammay + np.sqrt(0.5) * rnd_data(nexp, 1)

    data_exp = torch.cat((g(X), g(S2), g(S1), g(S3), Y, A), dim=1).numpy()
    pd.DataFrame(data_exp).to_csv(f"tmp3/exp_{i}.csv", index=False, header=False)

    alpha3 = rnd_norm_coef(1, dimS, np.sqrt(0.5))
    S1 = tau1
    S2 = tau2 + S1 @ alpha2
    S3 = tau3 + S2 @ alpha3
    Y = tauy + S3 @ alphay

    res_list.append(Y.numpy())

torch.save(res_list, "tmp3/result.pt")