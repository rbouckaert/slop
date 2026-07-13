package vb;

import java.util.Arrays;
import java.util.Random;

/**
 * Multivariate Variational Bayes using Stochastic Gradient Descent. Model: Data
 * ~ N(Theta, I), Prior Theta ~ N(0, 10^2 * I) Variational Distribution:
 * q(Theta) = N(mu, diag(sigma^2))
 */
public class VariationalBayesSGD {
	private final Random random = new Random();
	private final int dim;

	// Hyperparameters
	private final double priorStd = 10.0;
	private final double dataStd = 1.0;

	// Variational Parameters
	private final double[] mu;
	private final double[] logSigma;

	// SGD Settings
	private double learningRate = 0.01;
	private final int numSamples = 10;

	public VariationalBayesSGD(int dimension) {
	        this.dim = dimension;
	        this.mu = new double[dimension];
	        this.logSigma = new double[dimension];
	        // Initialize logSigma to 0 (sigma = 1)
	        Arrays.fill(logSigma, 0.0);
	    }

	/**
	 * Performs one step of SGD to maximize the ELBO.
	 * 
	 * @param data Mini-batch of data where each row is a D-dimensional observation.
	 */
	public void step(double[][] data) {
		double[] gradMu = new double[dim];
		double[] gradLogSigma = new double[dim];

		for (int s = 0; s < numSamples; s++) {
			// 1. Sample epsilon and transform to theta (Reparameterization Trick)
			double[] epsilon = new double[dim];
			double[] theta = new double[dim];
			for (int d = 0; d < dim; d++) {
				epsilon[d] = random.nextGaussian();
				double sigma = Math.exp(logSigma[d]);
				theta[d] = mu[d] + sigma * epsilon[d];
			}

			// 2. Compute Gradients for each dimension
			for (int d = 0; d < dim; d++) {
				double sigma = Math.exp(logSigma[d]);

				// Gradient of Log Prior: log p(theta_d)
				double dLogPrior_dTheta = -theta[d] / (priorStd * priorStd);

				// Gradient of Log Likelihood: sum log p(x_i_d | theta_d)
				double dLogLik_dTheta = 0;
				for (double[] row : data) {
					dLogLik_dTheta += (row[d] - theta[d]) / (dataStd * dataStd);
				}

				// Gradient of Log Variational: log q(theta_d)
				double dLogQ_dTheta = -(theta[d] - mu[d]) / (sigma * sigma);

				// Chain Rule: dELBO/dTheta
				double dELBO_dTheta = dLogPrior_dTheta + dLogLik_dTheta - dLogQ_dTheta;

				// Accumulate gradients for mu and logSigma
				gradMu[d] += dELBO_dTheta;
				// dTheta/dLogSigma = epsilon * sigma. Add 1 for the entropy term derivative.
				gradLogSigma[d] += (dELBO_dTheta * epsilon[d] * sigma) + 1;
			}
		}

		// 3. Update Variational Parameters (Gradient Ascent)
		for (int d = 0; d < dim; d++) {
			mu[d] += learningRate * (gradMu[d] / numSamples);
			logSigma[d] += learningRate * (gradLogSigma[d] / numSamples);
		}
	}

	public double[] getMeans() {
		return mu.clone();
	}

	public double[] getStds() {
		double[] stds = new double[dim];
		for (int i = 0; i < dim; i++)
			stds[i] = Math.exp(logSigma[i]);
		return stds;
	}

	public static void main(String[] args) {
		int dimension = 3;
		int numObservations = 200;
		double[] trueTheta = { 5.0, -2.0, 10.0 };

		// 1. Generate Toy Data
		double[][] data = new double[numObservations][dimension];
		Random r = new Random();
		for (int i = 0; i < numObservations; i++) {
			for (int d = 0; d < dimension; d++) {
				data[i][d] = trueTheta[d] + r.nextGaussian();
			}
		}

		// 2. Train
		VariationalBayesSGD vb = new VariationalBayesSGD(dimension);
		System.out.println("Training Multivariate VB...");
		for (int iter = 0; iter < 2000; iter++) {
			vb.step(data);
			if (iter % 500 == 0) {
				System.out.printf("Iter %d | Means: %s\n", iter, Arrays.toString(vb.getMeans()));
			}
		}

		// 3. Results
		System.out.println("\nFinal Results:");
		System.out.println("True Theta: " + Arrays.toString(trueTheta));
		System.out.println("Est. Means: " + Arrays.toString(vb.getMeans()));
		System.out.println("Est. Stds:  " + Arrays.toString(vb.getStds()));
	}
}