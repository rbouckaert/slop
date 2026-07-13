package vb;

import java.util.Arrays;
import java.util.Random;

public class VariationalBayesSGD2 {
    private final Random random = new Random();
    private final int dim;

    // Variational Parameters
    private final double[] mu;
    private final double[] logSigma;

    // Adam Optimizer State
    private final double[] mMu, vMu;
    private final double[] mLogSigma, vLogSigma;
    private final double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    private int t = 0; // Timestep for Adam

    // Settings
    private double learningRate = 0.005; 
    private final double gradClip = 5.0; // Prevent exploding gradients
    private final int numSamples = 8;

    public VariationalBayesSGD2(int dimension) {
        this.dim = dimension;
        this.mu = new double[dimension];
        this.logSigma = new double[dimension];
        
        this.mMu = new double[dimension];
        this.vMu = new double[dimension];
        this.mLogSigma = new double[dimension];
        this.vLogSigma = new double[dimension];
        
        // Initialize logSigma to a small positive value (log(1) = 0)
        Arrays.fill(logSigma, 0.0);
    }

    public void step(double[][] data) {
        t++;
        double[] gMu = new double[dim];
        double[] gLogSigma = new double[dim];

        for (int s = 0; s < numSamples; s++) {
            for (int d = 0; d < dim; d++) {
                double sigma = Math.exp(logSigma[d]);
                double eps = random.nextGaussian();
                double theta = mu[d] + sigma * eps;

                // 1. Gradients of log-prior p(theta) ~ N(0, 10^2)
                double dLogPrior = -theta / (10.0 * 10.0);

                // 2. Gradients of log-likelihood (Average over batch to stay scale-invariant)
                double dLogLik = 0;
                for (double[] row : data) {
                    dLogLik += (row[d] - theta); // Assuming data variance = 1
                }
                dLogLik /= data.length; 

                // 3. Gradient of log-variational q(theta)
                double dLogQ = -(theta - mu[d]) / (sigma * sigma + 1e-7);

                double dELBO_dTheta = dLogPrior + dLogLik - dLogQ;

                gMu[d] += dELBO_dTheta;
                gLogSigma[d] += (dELBO_dTheta * eps * sigma) + 1.0;
            }
        }

        // Average samples and Apply Adam Update
        for (int d = 0; d < dim; d++) {
            updateParam(d, mu, gMu[d] / numSamples, mMu, vMu);
            updateParam(d, logSigma, gLogSigma[d] / numSamples, mLogSigma, vLogSigma);
            
            // Numerical Safety: Clamp logSigma so sigma stays between ~0.0001 and 100
            logSigma[d] = Math.max(-10.0, Math.min(2.0, logSigma[d]));
        }
    }

    private void updateParam(int d, double[] param, double rawGrad, double[] m, double[] v) {
        // 1. Gradient Clipping
        double grad = Math.max(-gradClip, Math.min(gradClip, rawGrad));

        // 2. Adam Logic
        m[d] = beta1 * m[d] + (1 - beta1) * grad;
        v[d] = beta2 * v[d] + (1 - beta2) * (grad * grad);

        double mHat = m[d] / (1 - Math.pow(beta1, t));
        double vHat = v[d] / (1 - Math.pow(beta2, t));

        // 3. Apply Update (Gradient Ascent)
        param[d] += learningRate * mHat / (Math.sqrt(vHat) + epsilon);
        
        // Final NaN check
        if (Double.isNaN(param[d])) param[d] = 0; 
    }

    public double[] getMeans() { return mu.clone(); }

    public static void main(String[] args) {
        int dimension = 5;
        double[][] data = new double[100][dimension];
        Random r = new Random();
        double  alpha = 3;
        for (int i = 0; i < 100; i++) {
            data[i][0] = 5.0 + r.nextGaussian() * alpha; // True mean 5
            data[i][1] = -10.0 + r.nextGaussian() * alpha; // True mean -10
            for (int j = 2; j < dimension; j++) {
            	data[i][j] = j + r.nextGaussian() * alpha; // True mean -10
            }
        }

        VariationalBayesSGD2 vb = new VariationalBayesSGD2(dimension);
        for (int i = 0; i < 15000; i++) {
            vb.step(data);
            if (i % 1000 == 0) {
                System.out.println("Iteration " + i + ": " + Arrays.toString(vb.getMeans()));
            }
        }
        System.out.println("Final Estimates: " + Arrays.toString(vb.getMeans()));
        System.out.println("logSigma: " + Arrays.toString(vb.logSigma));
    }
}

