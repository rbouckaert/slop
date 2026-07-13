package vb;

import java.util.Arrays;
import java.util.Random;

public class CoordinateWiseNumericalVB {
    private final ProbabilisticModel model;
    private final int dim;
    private final Random random = new Random();

    // Variational Parameters
    private final double[] mu, logSigma;
    
    // Adam State (one for each parameter)
    private final double[] mMu, vMu, mLogSigma, vLogSigma;
    private int[] tMu, tLogSigma; // Individual timesteps for Adam

    // Hyperparameters
    private double learningRate = 0.005;
    private final double epsilonFD = 1e-5;
    private final double gradClip = 10.0;

    public CoordinateWiseNumericalVB(int dimension, ProbabilisticModel model) {
        this.dim = dimension;
        this.model = model;
        this.mu = new double[dim];
        this.logSigma = new double[dim];
        this.mMu = new double[dim];
        this.vMu = new double[dim];
        this.mLogSigma = new double[dim];
        this.vLogSigma = new double[dim];
        this.tMu = new int[dim];
        this.tLogSigma = new int[dim];
        Arrays.fill(logSigma, 0.0);
    }

    /**
     * Executes one "pass" over all dimensions, updating each coordinate immediately.
     */
    public void step(double[][] data) {
        // We draw a fresh epsilon vector for this pass
        double[] eps = new double[dim];
        for (int i = 0; i < dim; i++) eps[i] = random.nextGaussian();

        for (int d = 0; d < dim; d++) {
            // 1. Current theta based on latest parameters
            double[] currentTheta = getCurrentTheta(eps);
            double sigma = Math.exp(logSigma[d]);

            // 2. Numerical partial derivative for this dimension only
            double dLogJoint = estimatePartialDerivative(d, currentTheta, data);

            // 3. Calculate ELBO gradients for this dimension
            // dLogQ is the analytical gradient of the variational Gaussian
            double dLogQ = -(currentTheta[d] - mu[d]) / (sigma * sigma + 1e-8);
            double dELBO_dTheta = dLogJoint - dLogQ;

            double gMu = dELBO_dTheta;
            double gLogSigma = (dELBO_dTheta * eps[d] * sigma) + 1.0;

            // 4. Update this coordinate IMMEDIATELY
            updateParameter(d, mu, gMu, mMu, vMu, tMu);
            updateParameter(d, logSigma, gLogSigma, mLogSigma, vLogSigma, tLogSigma);

            // Safety: Clamp variance
            logSigma[d] = Math.max(-8.0, Math.min(4.0, logSigma[d]));
        }
    }

    private double[] getCurrentTheta(double[] eps) {
        double[] theta = new double[dim];
        for (int i = 0; i < dim; i++) {
            theta[i] = mu[i] + Math.exp(logSigma[i]) * eps[i];
        }
        return theta;
    }

    /**
     * Estimates partial derivative w.r.t theta[d] using Central Differences.
     */
    private double estimatePartialDerivative(int d, double[] theta, double[][] data) {
        double originalVal = theta[d];
        
        theta[d] = originalVal + epsilonFD;
        double fPlus = model.logJoint(theta, data);
        
        theta[d] = originalVal - epsilonFD;
        double fMinus = model.logJoint(theta, data);
        
        theta[d] = originalVal; // Restore
        return (fPlus - fMinus) / (2.0 * epsilonFD);
    }

    private void updateParameter(int d, double[] param, double rawGrad, double[] m, double[] v, int[] tArr) {
        tArr[d]++;
        double g = Math.max(-gradClip, Math.min(gradClip, rawGrad));
        
        // Adam update
        m[d] = 0.9 * m[d] + 0.1 * g;
        v[d] = 0.999 * v[d] + 0.001 * (g * g);
        
        double mHat = m[d] / (1.0 - Math.pow(0.9, tArr[d]));
        double vHat = v[d] / (1.0 - Math.pow(0.999, tArr[d]));
        
        param[d] += learningRate * mHat / (Math.sqrt(vHat) + 1e-8);
    }

    public double[] getMeans() { return mu.clone(); }
}