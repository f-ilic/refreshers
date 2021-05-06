import numpy as np
rng = np.random.RandomState(42)

def generate_data_gaussians(means, covariances, num_samples):
    num_classes = len(means)
    samples = np.zeros((num_samples * num_classes, 2))
    labels = np.zeros((num_samples * num_classes))

    for cls, (mean, cov) in enumerate(zip(means, covariances)):
        samples[cls * num_samples:(cls + 1) * num_samples, :] = rng.multivariate_normal(mean, cov, num_samples)
        labels[cls * num_samples:(cls + 1) * num_samples] = np.ones(num_samples) * cls

    return samples, labels.astype(np.int)


def generate_donut(mean, sigma, radius, num_samples):
    r = radius + np.random.normal(mean, sigma, num_samples)
    angle = np.random.rand(num_samples) * 2 * np.pi

    x = r * np.cos(angle)
    y = r * np.sin(angle)

    samples = np.vstack((x, y))
    labels = np.ones((num_samples))
    return samples.T, labels.astype(np.int)


def donut_dataset(num_samples=1000):
    s0, l0 = generate_donut(0, 2, 20, num_samples)
    s1, l1 = generate_data_gaussians([np.array([0, 0])], [np.array([[1, 0], [0, 1]])], num_samples)

    samples = np.vstack((s0, s1))
    labels = np.hstack((l0, l1))

    return samples.astype(np.float), labels.astype(np.long)


def simple_dataset(num_samples=1000):
    m1 = np.array([-6.0, 1.0])
    m2 = np.array([1.0, -1.0])
    means = [m1, m2]

    c1 = np.array([[1, 0], [0, 1]])
    c2 = np.array([[1, 0], [0, 1]])
    covariances = [c1, c2]
    samples, labels = generate_data_gaussians(means, covariances, num_samples)

    return samples.astype(np.float), labels.astype(np.long)

