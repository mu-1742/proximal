import numpy as np


class ProxCIData:
    def __init__(self, X, S1, S2, S3, Y, A):
        self.X = X
        self.S1 = S1
        self.S2 = S2
        self.S3 = S3
        self.Y = Y
        self.A = A
        self.n = X.shape[0]
        assert S1.shape[0] == self.n
        assert S2.shape[0] == self.n
        assert S3.shape[0] == self.n
        assert Y.shape[0] == self.n
        assert A.shape[0] == self.n

    @property
    def Z(self):
        # return np.hstack((unvec(self.S1), unvec(self.S2)))
        return np.hstack(unvec(self.S1))

    @property
    def W(self):
        # return np.hstack((unvec(self.S2), unvec(self.S3)))
        return np.hstack(unvec(self.S3))

    @property
    def r1(self):
        return np.hstack((unvec(self.Z), unvec(self.X), unvec(self.S2)))

    @property
    def r2(self):
        return np.hstack((unvec(self.W), unvec(self.X), unvec(self.S2)))

    def create_crossfit_split(self, n_splits, cache=True):
        """split the dataset into n_split groups for cross-fitting"""
        if n_splits <= 1:
            idx = np.arange(self.n)
            split_indices = [{"train": idx, "eval": idx}]
        else:
            idx = np.arange(self.n)
            np.random.shuffle(idx)
            split_size = int(np.floor(self.n / n_splits))
            split_indices = []
            for i in range(n_splits):
                start = i * split_size
                end = (i + 1) * split_size
                if i == n_splits - 1 and end < self.n - 1:
                    end = self.n - 1
                eval_idx = np.sort(idx[start:end])
                train_idx = np.sort(np.hstack((idx[: max(0, start)], idx[end:])))
                assert len(eval_idx) + len(train_idx) == self.n
                split_indices.append({"train": train_idx, "eval": eval_idx})

        if cache:
            self.split_indices = split_indices

        return split_indices


# utility functions
def unvec(a):
    return np.expand_dims(a, 1) if a.ndim == 1 else a


def join_data(*args):
    return np.hstack([unvec(a) for a in args])
