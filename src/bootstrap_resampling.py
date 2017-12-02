import numpy as np

sample = np.random.randint(low=1, high=100, size=10)
print('Original sample', sample)
print('Original Sample Mean', sample.mean())

# bootstrap re-sample 100 times by re-sampling with replacement fro the original sample
resample = [np.random.choice(sample, size=sample.shape) for i in range(100)]
print('Number of bootstrap re-samples', len(resample))
print('Example resample', resample[0])
resample_means = np.array([sample.mean() for sample in resample])
print('Mean of resamples\' mean', resample_means.mean())
