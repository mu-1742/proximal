import os
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


def load_data(num_simulations=2, data_dir="tmp2"):
    observational_data = []
    experimental_data = []
    for i in range(1, num_simulations + 1):
        obs_data = generate_data_from_file(os.path.join(data_dir, f"obs_{i}.csv"))
        exp_data = generate_data_from_file(os.path.join(data_dir, f"exp_{i}.csv"))
        observational_data.append(obs_data)
        experimental_data.append(exp_data)
    return observational_data, experimental_data


if __name__ == "__main__":
    observational_data, experimental_data = load_data(num_simulations=2)

    obs_Y = [data[4] for data in observational_data]
    exp_Y = [data[4] for data in experimental_data]
    obs_Y_mean = np.mean(np.concatenate(obs_Y))
    exp_Y_mean = np.mean(np.concatenate(exp_Y))

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


    print(f"obs_Y:{obs_Y_mean}")
    print(f"exp_Y:{exp_Y_mean}")
    print(f"OTC: {tau_OTC}")
    print(f"SEL: {tau_SEL}")
    print(f"DR: {tau_DR}")


