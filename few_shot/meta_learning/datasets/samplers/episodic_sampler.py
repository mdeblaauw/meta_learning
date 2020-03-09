import numpy as np
import random
from .base_sampler import BaseSampler


class EpisodicSampler(BaseSampler):
    """Episode sampler for the episodic_dataset class. It generates batches of
    n-shot, k-way, q-query tasks. Each n-shot task contains a "support set" of
    `k` sets of `n` samples and a "query set" of `k` sets of `q` samples.
    The support set and the query set are all grouped into one array, such
    that the first n * k samples are from the support set while the remaining
    q * k samples are from the query set. The support and query sets are
    sampled such that they are disjoint, i.e. they do not contain overlapping
    samples.
    """
    def __init__(self, dataset, configuration: Dict):
        """Initialisation of parameters.

        Arguments:
            dataset {torch.utils.data.Dataset} -- Should be subclass of
                episodic_dataset.py.
            configuration {Dict} -- Sampler configuration file.
        """
        super().__init__(dataset, configuration)
        self.episodes_per_epoch = self.configuration['episodes_per_epoch']
        self.k = self.configuration['k']
        self.n = self.configuration['n']
        self.q = self.configuration['q']

    def __len__(self) -> int:
        """The number of episodes per epoch.

        Returns:
            int -- Number of episodes per epoch.
        """
        return self.episodes_per_epoch

    def __iter__(self):
        """Iterator of batches per epoch of support and query sets.

        Yields:
            np.ndarray -- Numpy array with ids from dataset. For which
                the first n*k ids are support samples and the other q*k ids
                are query samples.
        """
        for _ in range(self.episodes_per_epoch):
            batch = []

            episode_classes = np.random.choice(
                self.dataset.df['class_id'].unique(),
                size=self.k, replace=False
            )

            chosen_class_df = self.dataset.df[
                self.dataset.df['class_id'].isin(episode_classes)
            ]

            support_k = {k: None for k in episode_classes}
            # Select n*k support samples first
            for k in episode_classes:
                support = df[df['class_id'] == k].sample(self.n)
                support_k[k] = support

                # Store sample ids
                for i, s in support.iterrows():
                    batch.append(s['id'])

            # Select n*q queries that are mutually exclusive from support set
            for k in episode_classes:
                query = df[
                    (
                        df['class_id'] == k
                    ) & (
                        ~df['id'].isin(support_k[k]['id'])
                    )
                ].sample(self.q)

                # Store sample ids
                for i, q in query.iterrows():
                    batch.append(q['id'])

            yield np.stack(batch)
