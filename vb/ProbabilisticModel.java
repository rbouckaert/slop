package vb;

public interface ProbabilisticModel {
    /**
     * Calculates the Log-Joint Probability: log p(Data, Theta)
     * This is log(Likelihood * Prior).
     * @param theta Current sample of latent variables
     * @param data The observed data batch
     * @return The log-probability value
     */
    double logJoint(double[] theta, double[][] data);
}