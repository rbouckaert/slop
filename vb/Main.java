package vb;

import java.util.Arrays;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        // Simple Model: log p(y | x, alpha, beta) + log p(alpha, beta)
        ProbabilisticModel myModel = (theta, data) -> {
            double alpha = theta[0];
            double beta = theta[1];
            double gamma = theta[2];
            double logProb = 0;

            // Log Priors (Normal(0, 10))
            logProb += -0.5 * (alpha * alpha) / 100.0;
            logProb += -0.5 * (beta * beta) / 100.0;

            // Log Likelihood
            for (double[] point : data) {
                double x = point[0];
                double y = point[1];
                double z = point[2];
                double prediction = alpha + beta * x;
                double prediction2 = gamma * y;
                
                // Gaussian Likelihood with noise std = 1.0
                logProb += -0.5 * Math.pow(y - prediction, 2) -0.5 * Math.pow(z - prediction2, 2);
            }
            return logProb / data.length; // Normalize by batch size for stability
        };

        // Create Toy Data (y = 2x + 5)
     // Create Toy Data (y = 2x + 5)
        double[][] data = new double[50][3];
        for (int i = 0; i < 50; i++) {
            data[i][0] = i * 0.1;
            data[i][1] = 5.0 + 2.0 * data[i][0] + new Random().nextGaussian() * 0.1;
            data[i][2] = 0.0 + data[i][1] + new Random().nextGaussian() * 0.1;
        }

        CoordinateWiseNumericalVB vb = new CoordinateWiseNumericalVB(3, myModel);
        for (int i = 0; i <= 10000; i++) {
            vb.step(data);
            if (i % 1000 == 0) System.out.println("Iteration " + i + ": " + Arrays.toString(vb.getMeans()));
        }//        int dimension = 2;
//        double[][] data = new double[100][dimension];
//        Random r = new Random();
//        double  alpha = 0.1;
//        for (int i = 0; i < 100; i++) {
//            data[i][0] = 5.0 + r.nextGaussian() * alpha; // True mean 5
//            data[i][1] = -10.0 + r.nextGaussian() * alpha; // True mean -10
//            for (int j = 2; j < dimension; j++) {
//            	data[i][j] = j + r.nextGaussian() * alpha; // True mean -10
//            }
//        }
//
//        NumericalBlackBoxVB vb = new NumericalBlackBoxVB(dimension, myModel);
//        for (int i = 0; i < 50000; i++) {
//            vb.step(data);
//            if (i % 10000 == 0) System.out.println("Iteration " + i + ": " + Arrays.toString(vb.getMeans()));
//        }
    }
}