from .proxci_dataset import *
from .minimax import *

class ProximalInference:
    def __init__(
        self,
        proxci_dataset_exp,
        proxci_dataset_obs=None,
        crossfit_folds=1,
        lambdas=None,
        gammas=None,
        cv=2,
        n_jobs=1,
        verbose=0,
        print_best_params=False,
    ):
        assert crossfit_folds >= 1
        self.crossfit_folds = crossfit_folds
        self.data_exp = proxci_dataset_exp
        self.data_obs = proxci_dataset_obs

        # minimax and gridsearch parameters
        self.hdim = self.data_exp.r1.shape[1]
        self.fdim = self.data_exp.r2.shape[1]
        self.lambdas = lambdas
        self.gammas = gammas
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.print_best_params = print_best_params

        self.cf_inds_exp = proxci_dataset_exp.create_crossfit_split(crossfit_folds)

        if self.data_obs:
            self.cf_inds_obs = proxci_dataset_obs.create_crossfit_split(crossfit_folds)
        else:
            self.cf_inds_obs = None

        self.h = [
            {a: self.estimate_h(a, fold=i, data_type="exp") for a in range(2)}
            for i in range(crossfit_folds)
        ]
        self.q = [
            {a: self.estimate_q(a, fold=i, data_type="obs") for a in range(2)}
            for i in range(crossfit_folds)
        ]

    def estimate_h(self, a, fold=0, data_type="exp"):
        data = self.data_exp if data_type == "exp" else self.data_obs
        cf_inds = self.cf_inds_exp if data_type == "exp" else self.cf_inds_obs

        g1 = -1 * (data.A == a)
        g2 = data.Y * (data.A == a)
        r1 = data.r1[cf_inds[fold]["train"]]
        r2 = data.r2[cf_inds[fold]["train"]]
        g1 = g1[cf_inds[fold]["train"]]
        g2 = g2[cf_inds[fold]["train"]]

        data = join_data(r1, r2, g1, g2)
        search = MinimaxRKHSCV(
            r1.shape[1],
            r2.shape[1],
            lambdas=self.lambdas,
            gammas=self.gammas,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        search.fit(data)
        if self.print_best_params > 0:
            print(f"h, a={a}", search.best_params_, "best score: ", search.best_score_)
        return search.best_estimator_.h_

    def estimate_q(self, a, fold=0, data_type="obs"):
        data = self.data_obs if data_type == "obs" else self.data_exp
        cf_inds = self.cf_inds_obs if data_type == "obs" else self.cf_inds_exp

        g1 = -1 * (data.A == a)
        g2 = np.ones(len(data.Y))
        r1 = data.r1[cf_inds[fold]["train"]]
        r2 = data.r2[cf_inds[fold]["train"]]
        g1 = g1[cf_inds[fold]["train"]]
        g2 = g2[cf_inds[fold]["train"]]

        data = join_data(r2, r1, g1, g2)
        search = MinimaxRKHSCV(
            r1.shape[1],
            r2.shape[1],
            lambdas=self.lambdas,
            gammas=self.gammas,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        search.fit(data)
        if self.print_best_params > 0:
            print(f"q, a={a}", search.best_params_, "best score: ", search.best_score_)
        return search.best_estimator_.h_

    def por(self, reduction=np.mean):
        """Estimator based on function h"""
        estimates = []
        for fold in range(self.crossfit_folds):
            r = self.data_exp.r1[self.cf_inds_exp[fold]["eval"]]
            estimates = np.append(estimates, self.h[fold][1](r) - self.h[fold][0](r))
        return estimates if reduction is None else reduction(estimates)

    def pipw(self, reduction=np.mean):
        """Estimator based on function q"""
        estimates = []
        for fold in range(self.crossfit_folds):
            fold_idx = self.cf_inds_obs[fold]["eval"]
            I0 = self.data_obs.A[fold_idx] == 0
            I1 = self.data_obs.A[fold_idx] == 1
            r = self.data_obs.r2[fold_idx]
            y = self.data_obs.Y[fold_idx]
            est = I1 * y * self.q[fold][1](r) - I0 * y * self.q[fold][0](r)
            estimates = np.append(estimates, est)
        return estimates if reduction is None else reduction(estimates)

    def dr(self, reduction=np.mean):
        """Doubly robust estimator"""
        estimates = []
        for fold in range(self.crossfit_folds):
            fold_idx_exp = self.cf_inds_exp[fold]["eval"]
            fold_idx_obs = self.cf_inds_obs[fold]["eval"]

            r1_exp = self.data_exp.r1[fold_idx_exp]
            r2_obs = self.data_obs.r2[fold_idx_obs]
            y_exp = self.data_exp.Y[fold_idx_exp]

            h0_exp = self.h[fold][0](r1_exp)
            h1_exp = self.h[fold][1](r1_exp)
            q0_obs = self.q[fold][0](r2_obs)
            q1_obs = self.q[fold][1](r2_obs)

            est = (y_exp - h1_exp) * q1_obs + h1_exp - (y_exp - h0_exp) * q0_obs + h0_exp
            estimates = np.append(estimates, est)
        return estimates if reduction is None else reduction(estimates)
