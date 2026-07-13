package vb;


import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.util.FastMath;

public class VariationalBayesSGD3 {

    // Hyperparameters
    private static final double LEARNING_RATE = 0.01;
    private static final int NUM_ITERATIONS = 1000;
    private static final int BATCH_SIZE = 32;

    // Prior parameters (Gaussian)
    private static final double PRIOR_MEAN = 0.0;
    private static final double PRIOR_VARIANCE = 1.0;

    // Data parameters
    private static final int N = 100; // Number of data points
    private static final int D = 1;   // Number of features

    public static void main(String[] args) {
        RandomGenerator rng = new JDKRandomGenerator();
        rng.setSeed(42);

        // Generate synthetic data: y = 2*x + noise
        RealMatrix X = generateRandomMatrix(N, D, rng);
        RealMatrix y = X.scalarMultiply(2.0).add(generateRandomMatrix(N, 1, rng).scalarMultiply(0.5));

        // Variational parameters (mean and log variance for each weight)
        RealMatrix mu = MatrixUtils.createRealMatrix(D, 1);
        RealMatrix logVar = MatrixUtils.createRealMatrix(D, 1);

        // SGD loop
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            // Random batch indices
            int[] indices = new int[BATCH_SIZE];
            for (int i = 0; i < BATCH_SIZE; i++) {
                indices[i] = rng.nextInt(N);
            }

            // Extract batch
            RealMatrix X_batch = getRows(X, indices);
            RealMatrix y_batch = getRows(y, indices);

            // Compute gradients
            RealMatrix[] gradients = computeGradients(X_batch, y_batch, mu, logVar, rng);

            // Update parameters
            mu = mu.add(gradients[0].scalarMultiply(LEARNING_RATE));
            logVar = logVar.add(gradients[1].scalarMultiply(LEARNING_RATE));

            if (iter % 100 == 0) {
                System.out.printf("Iter %d: mu = %.4f, logVar = %.4f%n",
                    iter, mu.getEntry(0, 0), logVar.getEntry(0, 0));
            }
        }

        System.out.println("\nFinal variational parameters:");
        System.out.println("mu = " + mu.getEntry(0, 0));
        System.out.println("sigma^2 = " + FastMath.exp(logVar.getEntry(0, 0)));
    }

    // Generate random matrix
    private static RealMatrix generateRandomMatrix(int rows, int cols, RandomGenerator rng) {
        double[][] data = new double[rows][cols];
        NormalDistribution nd = new NormalDistribution(rng, 0, 1);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = nd.sample();
            }
        }
        return MatrixUtils.createRealMatrix(data);
    }

    // Get rows from matrix
    private static RealMatrix getRows(RealMatrix matrix, int[] indices) {
        double[][] data = new double[indices.length][matrix.getColumnDimension()];
        for (int i = 0; i < indices.length; i++) {
            data[i] = matrix.getRow(indices[i]);
        }
        return MatrixUtils.createRealMatrix(data);
    }

    // Element-wise square root of a matrix
    private static RealMatrix sqrt(RealMatrix matrix) {
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = FastMath.sqrt(matrix.getEntry(i, j));
            }
        }
        return MatrixUtils.createRealMatrix(data);
    }

    // Compute gradients for mu and logVar
    private static RealMatrix[] computeGradients(
            RealMatrix X, RealMatrix y, RealMatrix mu, RealMatrix logVar, RandomGenerator rng) {
        int n = X.getRowDimension();
        RealMatrix sigma2 = new Array2DRowRealMatrix(logVar.getRowDimension(), logVar.getColumnDimension());
        for (int i = 0; i < sigma2.getRowDimension(); i++) {
            for (int j = 0; j < sigma2.getColumnDimension(); j++) {
                sigma2.setEntry(i, j, FastMath.exp(logVar.getEntry(i, j)));
            }
        }

        // Sample epsilon ~ N(0,1)
        RealMatrix epsilon = generateRandomMatrix(n, 1, rng);

        // Reparameterization trick: z = mu + sigma * epsilon
        RealMatrix sigma = sqrt(sigma2);
        RealMatrix z = mu.add(sigma.multiply(epsilon));

        // Predicted y
        RealMatrix y_pred = X.multiply(z);

        // ELBO gradient for mu
        RealMatrix grad_mu = X.transpose().multiply(y_pred.subtract(y)).scalarMultiply(-1.0 / n)
                .add(mu.scalarMultiply(-1.0 / PRIOR_VARIANCE));

        // ELBO gradient for logVar
        RealMatrix term1 = sigma.multiply(epsilon.transpose())
                .multiply(y_pred.subtract(y)).scalarMultiply(-1.0 / n);
        RealMatrix term2 = MatrixUtils.createRealMatrix(new double[][]{{0.5 * (1 - 1 / sigma2.getEntry(0, 0) - logVar.getEntry(0, 0))}});
        RealMatrix grad_logVar = term1.add(term2);

        return new RealMatrix[]{grad_mu, grad_logVar};
    }
}