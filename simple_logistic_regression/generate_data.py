import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(42)


def generate_data_gaussians(means, covariances, num_samples, num_classes):
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


def generate_donut_and_gaussian(num_samples=1000):
    s0, l0 = generate_donut(0, 0.3, 10, num_samples)
    s1, l1 = generate_data_gaussians([np.array([0,0])], [np.array([[1, 0], [0, 1]])], num_samples, 1)

    samples = np.vstack((s0, s1))
    labels = np.hstack((l0, l1))

    for cls in np.unique(labels):
        c = samples[labels == cls]
        # plt.scatter(c[:, 0], c[:, 1])

    return samples, labels


def multiple_blobs_dataset():
    m1 = np.array([-5.0, 2.0])
    m2 = np.array([1.0, 3.0])
    m3 = np.array([2.0, -2.0])
    means = [m1, m2]#, m3]

    c1 = np.array([[1, 0], [0, 2]])
    c2 = np.array([[3, 0], [0, 0.5]])
    c3 = np.array([[1, 0], [0, 6]])
    covariances = [c1, c2]#, c3]

    num_samples = 100
    num_classes = 3

    samples, labels = generate_data_gaussians(means, covariances, num_samples, num_classes)

    for cls in range(num_classes):
        c = samples[labels == cls]
        # plt.scatter(c[:, 0], c[:, 1])
    return samples, labels
    # plt.show()


def donut_dataset():
    samples, labels = generate_donut(0, 0.1, 1, 1000)
    plt.scatter(samples[:, 0], samples[:, 1])
    # plt.show()


if __name__ == '__main__':
    generate_donut_and_gaussian()
