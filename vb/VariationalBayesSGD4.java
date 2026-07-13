package vb;

import java.util.Random;

/**
 * A simple implementation of Variational Bayes using Stochastic Gradient Descent.
 * Model: Data ~ N(theta, 1), Prior theta ~ N(0, 10).
 * Goal: Find variational posterior q(theta) = N(mu, sigma^2) that approximates p(theta | data).
 */
public class VariationalBayesSGD4 {
    private final Random random = new Random();
    
    // Hyperparameters
    private final double priorMean = 0.0;
    private final double priorStd = 10.0;
    private final double dataStd = 1.0;
    
    // Variational Parameters (to be learned)
    private double mu = 0.0;
    private double logSigma = 0.0; // We optimize log(sigma) to ensure sigma > 0
    
    // SGD Settings
    private double learningRate = 0.01;
    private final int numSamples = 10; // Samples for MC estimation of ELBO gradient

    /**
     * Performs one step of Stochastic Gradient Descent on the ELBO.
     * @param data Mini-batch of observed data points
     */
    public void step(double[] data) {
        double gradMu = 0;
        double gradLogSigma = 0;

        for (int i = 0; i < numSamples; i++) {
            double eps = random.nextGaussian();
            double sigma = Math.exp(logSigma);
            double theta = mu + sigma * eps;

            // 1. Log Prior: log p(theta)
            // d/d_theta (-0.5 * (theta - m0)^2 / s0^2) = -(theta - m0) / s0^2
            double dLogPrior_dTheta = -(theta - priorMean) / (priorStd * priorStd);

            // 2. Log Likelihood: sum log p(x | theta)
            // d/d_theta (-0.5 * (x - theta)^2 / sd^2) = (x - theta) / sd^2
            double dLogLik_dTheta = 0;
            for (double x : data) {
                dLogLik_dTheta += (x - theta) / (dataStd * dataStd);
            }

            // 3. Log Variational: log q(theta)
            // d/d_theta (-0.5 * (theta - mu)^2 / sigma^2) = -(theta - mu) / sigma^2
            double dLogQ_dTheta = -(theta - mu) / (sigma * sigma);

            // Combine gradients using chain rule from reparameterization
            // dELBO/dTheta = dLogPrior/dTheta + dLogLik/dTheta - dLogQ/dTheta
            double dELBO_dTheta = dLogPrior_dTheta + dLogLik_dTheta - dLogQ_dTheta;

            // dTheta/dMu = 1
            gradMu += dELBO_dTheta;
            // dTheta/dLogSigma = dTheta/dSigma * dSigma/dLogSigma = eps * sigma
            gradLogSigma += dELBO_dTheta * (eps * sigma) + 1; // +1 comes from the Entropy term derivative d(log sigma)/d(log sigma)
        }

        // Update parameters (Gradient Ascent because we maximize ELBO)
        mu += learningRate * (gradMu / numSamples);
        logSigma += learningRate * (gradLogSigma / numSamples);
    }

    public double getPosteriorMean() { return mu; }
    public double getPosteriorStd() { return Math.exp(logSigma); }

    public static void main(String[] args) {
        // 1. Generate Toy Data (True theta = 5.0)
        double trueTheta = 5.0;
        double[] data = new double[100];
        Random r = new Random();
        for (int i = 0; i < 100; i++) {
        	data[i] = trueTheta + r.nextGaussian();
        }

        // 2. Initialize and Train
        VariationalBayesSGD4 vb = new VariationalBayesSGD4();
        System.out.println("Starting Training...");
        for (int iter = 0; iter < 1000; iter++) {
            vb.step(data);
            if (iter % 200 == 0) {
                System.out.printf("Iteration %d: Mean = %.4f, Std = %.4f\n", 
                                  iter, vb.getPosteriorMean(), vb.getPosteriorStd());
            }
        }

        // 3. Final Result
        System.out.println("\nFinal Variational Posterior:");
        System.out.printf("Mean: %.4f (Analytical target ~5.0)\n", vb.getPosteriorMean());
        System.out.printf("Std:  %.4f\n", vb.getPosteriorStd());
        System.out.printf("logSigma:  %.4f\n", vb.logSigma);
    }
}