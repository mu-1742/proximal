import os
import torch
import numpy as np
import pandas as pd
from proxci.proxci_dataset import ProxCIData
from proxci.proximal_inference import ProximalInference
import warnings
warnings.filterwarnings("ignore")

def generate_data_from_file(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :10].values
    S2 = data.iloc[:, 10:15].values
    S1 = data.iloc[:, 15:20].values
    S3 = data.iloc[:, 20:25].values
    Y = data.iloc[:, 25].values
    A = data.iloc[:, 26].values
    return X, S2, S1, S3, Y, A


def load_data(num_simulations=10, data_dir="tmp3"):
    observational_data = []
    experimental_data = []
    for i in range(1, num_simulations + 1):
        obs_data = generate_data_from_file(os.path.join(data_dir, f"obs_{i}.csv"))
        exp_data = generate_data_from_file(os.path.join(data_dir, f"exp_{i}.csv"))
        observational_data.append(obs_data)
        experimental_data.append(exp_data)
    return observational_data, experimental_data


if __name__ == "__main__":
    observational_data, experimental_data = load_data(num_simulations=10)

    Y = np.concatenate([data[4] for data in experimental_data])
    A = np.concatenate([data[5] for data in experimental_data])

    mean_Y1 = Y[A == 1].mean()
    mean_Y0 = Y[A == 0].mean()
    tau_Y = mean_Y1 - mean_Y0

    proxci_dataset_obs = ProxCIData(*observational_data[0])
    proxci_dataset_exp = ProxCIData(*experimental_data[0])

    proximal_inference = ProximalInference(
        proxci_dataset_exp,
        proxci_dataset_obs,
        crossfit_folds=5,
        lambdas=[0.1, 1.0, 10.0],
        gammas=[0.1, 1.0, 10.0],
        cv=5,
        n_jobs=1,
        verbose=0,
        print_best_params=True
    )

    tau_OTC = proximal_inference.por(reduction=np.mean)
    tau_SEL = proximal_inference.pipw(reduction=np.mean)
    tau_DR = proximal_inference.dr(reduction=np.mean)

    mae_OTC = np.mean(np.abs(tau_Y - tau_OTC))
    mae_DR = np.mean(np.abs(tau_Y - tau_DR))

    print(f"tau_Y:{tau_Y}")
    print(f"tau_OTC: {tau_OTC}, \nMAE_OTC: {mae_OTC}")
    print(f"tau_SEL: {tau_SEL}")
    print(f"tau_DR: {tau_DR}, \nMAE_DR: {mae_DR}")


