package vb;

import java.util.Arrays;
import java.util.Random;

public class NumericalBlackBoxVB {
    private final ProbabilisticModel model;
    private final int dim;
    private final Random random = new Random();

    // Variational Parameters & Adam State
    private final double[] mu, logSigma;
    private final double[] mMu, vMu, mLogSigma, vLogSigma;
    private int t = 0;

    // Hyperparameters
    private double learningRate = 0.005;
    private final double epsilonFD = 1e-5; // Finite Difference step size
    private final double gradClip = 10.0;
    private final int numSamples = 5; // Samples to estimate ELBO gradient

    public NumericalBlackBoxVB(int dimension, ProbabilisticModel model) {
        this.dim = dimension;
        this.model = model;
        this.mu = new double[dim];
        this.logSigma = new double[dim];
        this.mMu = new double[dim];
        this.vMu = new double[dim];
        this.mLogSigma = new double[dim];
        this.vLogSigma = new double[dim];
        Arrays.fill(logSigma, 0.0); // sigma starts at 1.0
    }

    public void step(double[][] data) {
        t++;
        double[] gMu = new double[dim];
        double[] gLogSigma = new double[dim];

        for (int s = 0; s < numSamples; s++) {
            // 1. Reparameterization: theta = mu + exp(logSigma) * eps
            double[] eps = new double[dim];
            double[] theta = new double[dim];
            for (int d = 0; d < dim; d++) {
                eps[d] = random.nextGaussian();
                theta[d] = mu[d] + Math.exp(logSigma[d]) * eps[d];
            }

            // 2. Numerical Gradient of the Model: grad_theta log p(x, theta)
            double[] gradLogJoint = estimateNumericalGradient(theta, data);

            // 3. Compute ELBO Gradients for each dimension
            for (int d = 0; d < dim; d++) {
                double sigma = Math.exp(logSigma[d]);
                
                // Analytical gradient of log q(theta) for a Gaussian: -(theta - mu)/sigma^2
                double dLogQ = -(theta[d] - mu[d]) / (sigma * sigma + 1e-8);

                // Total gradient with respect to theta
                double dELBO_dTheta = gradLogJoint[d] - dLogQ;

                // Map back to variational parameters using chain rule
                gMu[d] += dELBO_dTheta;
                gLogSigma[d] += (dELBO_dTheta * eps[d] * sigma) + 1.0; // +1 from entropy
            }
        }

        // 4. Update via Adam
        for (int d = 0; d < dim; d++) {
            updateAdam(d, mu, gMu[d] / numSamples, mMu, vMu);
            updateAdam(d, logSigma, gLogSigma[d] / numSamples, mLogSigma, vLogSigma);
            logSigma[d] = Math.max(-8.0, Math.min(4.0, logSigma[d])); // Bound variance
        }
    }

    /**
     * Estimates the gradient of logJoint w.r.t Theta using Central Differences.
     */
    private double[] estimateNumericalGradient(double[] theta, double[][] data) {
        double[] grad = new double[dim];
        double[] tempTheta = theta.clone();

        for (int i = 0; i < dim; i++) {
            double originalVal = tempTheta[i];
            
            tempTheta[i] = originalVal + epsilonFD;
            double fPlus = model.logJoint(tempTheta, data);
            
            tempTheta[i] = originalVal - epsilonFD;
            double fMinus = model.logJoint(tempTheta, data);
            
            grad[i] = (fPlus - fMinus) / (2.0 * epsilonFD);
            tempTheta[i] = originalVal; // Reset for next dimension
        }
        return grad;
    }

    private void updateAdam(int d, double[] param, double rawGrad, double[] m, double[] v) {
        double g = Math.max(-gradClip, Math.min(gradClip, rawGrad));
        m[d] = 0.9 * m[d] + 0.1 * g;
        v[d] = 0.999 * v[d] + 0.001 * (g * g);
        double mHat = m[d] / (1.0 - Math.pow(0.9, t));
        double vHat = v[d] / (1.0 - Math.pow(0.999, t));
        param[d] += learningRate * mHat / (Math.sqrt(vHat) + 1e-8);
    }

    public double[] getMeans() { return mu; }
}
