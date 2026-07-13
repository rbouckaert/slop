package vb;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

public class ParallelBlockVB {
    private final ProbabilisticModel model;
    private final int dim;
    private final int blockSize;
    private final ExecutorService executor;
    private final Random random = new Random();

    // Variational Parameters & Adam State
    private final double[] mu, logSigma;
    private final double[] mMu, vMu, mLogSigma, vLogSigma;
    private final int[] tParam;

    // Hyperparameters
    private final double learningRate = 0.005;
    private final double epsilonFD = 1e-5;
    private final double gradClip = 5.0;

    public ParallelBlockVB(int dimension, int blockSize, int numThreads, ProbabilisticModel model) {
        this.dim = dimension;
        this.model = model;
        this.blockSize = blockSize;
        this.executor = Executors.newFixedThreadPool(numThreads);
        
        this.mu = new double[dim];
        this.logSigma = new double[dim];
        this.mMu = new double[dim];
        this.vMu = new double[dim];
        this.mLogSigma = new double[dim];
        this.vLogSigma = new double[dim];
        this.tParam = new int[dim];
        Arrays.fill(logSigma, 0.0);
    }

    public void step(double[][] data) throws InterruptedException, ExecutionException {
        // 1. Prepare global epsilon for the reparameterization trick
        double[] eps = new double[dim];
        for (int i = 0; i < dim; i++) eps[i] = random.nextGaussian();

        // 2. Iterate through blocks
        for (int blockStart = 0; blockStart < dim; blockStart += blockSize) {
            int blockEnd = Math.min(blockStart + blockSize, dim);
            
            // Current theta sample (calculated once per block to use freshest mu/sigma)
            double[] theta = new double[dim];
            for (int i = 0; i < dim; i++) theta[i] = mu[i] + Math.exp(logSigma[i]) * eps[i];

            // 3. Calculate gradients for the block in parallel
            List<Future<GradientResult>> futures = new ArrayList<>();
            for (int d = blockStart; d < blockEnd; d++) {
                final int dimensionIndex = d;
                final double dimensionEps = eps[d];
                // Each thread works on a copy of theta to ensure thread safety during perturbation
                final double[] thetaCopy = theta.clone(); 
                
                futures.add(executor.submit(() -> calculateDimGradient(dimensionIndex, dimensionEps, thetaCopy, data)));
            }

            // 4. Collect results and update parameters for this block
            for (Future<GradientResult> future : futures) {
                GradientResult res = future.get();
                updateAdam(res.index, res.gMu, res.gLogSigma);
                // Clamp variance for stability
                logSigma[res.index] = Math.max(-8.0, Math.min(4.0, logSigma[res.index]));
            }
        }
    }

    private GradientResult calculateDimGradient(int d, double eps_d, double[] theta, double[][] data) {
        double sigma = Math.exp(logSigma[d]);
        
        // Numerical Partial Derivative: d/dTheta log p(x, theta)
        double originalTheta = theta[d];
        theta[d] = originalTheta + epsilonFD;
        double fPlus = model.logJoint(theta, data);
        theta[d] = originalTheta - epsilonFD;
        double fMinus = model.logJoint(theta, data);
        theta[d] = originalTheta;

        double dLogJoint = (fPlus - fMinus) / (2.0 * epsilonFD);

        // Analytical Gradient: d/dTheta log q(theta)
        double dLogQ = -(theta[d] - mu[d]) / (sigma * sigma + 1e-8);

        // Chain Rule for Reparameterization
        double dELBO_dTheta = dLogJoint - dLogQ;
        double gMu = dELBO_dTheta;
        double gLogSigma = (dELBO_dTheta * eps_d * sigma) + 1.0;

        return new GradientResult(d, gMu, gLogSigma);
    }

    private void updateAdam(int d, double gMu, double gLogSigma) {
        tParam[d]++;
        updateSingleParam(d, mu, gMu, mMu, vMu);
        updateSingleParam(d, logSigma, gLogSigma, mLogSigma, vLogSigma);
    }

    private void updateSingleParam(int d, double[] param, double rawGrad, double[] m, double[] v) {
        double g = Math.max(-gradClip, Math.min(gradClip, rawGrad));
        m[d] = 0.9 * m[d] + 0.1 * g;
        v[d] = 0.999 * v[d] + 0.001 * (g * g);
        double mHat = m[d] / (1.0 - Math.pow(0.9, tParam[d]));
        double vHat = v[d] / (1.0 - Math.pow(0.999, tParam[d]));
        param[d] += learningRate * mHat / (Math.sqrt(vHat) + 1e-8);
    }

    public void shutdown() { executor.shutdown(); }

    public double[] getMeans() { return mu.clone(); }

    // Helper class to hold computed gradients
    private static class GradientResult {
        int index;
        double gMu, gLogSigma;
        GradientResult(int i, double gm, double gls) { index = i; gMu = gm; gLogSigma = gls; }
    }
}