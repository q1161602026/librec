package net.librec.math.algorithm;

/**
 * This is a class implementing kernel smoothing functions
 * used in Local Low-Rank Matrix Approximation (LLORMA).
 *
 * @author Joonseok Lee
 * @version 1.2
 * @since 2013. 6. 11
 */
public class KernelSmoothing {
    public final static int TRIANGULAR_KERNEL = 201;
    public final static int UNIFORM_KERNEL = 202;
    public final static int EPANECHNIKOV_KERNEL = 203;
    public final static int GAUSSIAN_KERNEL = 204;

    public static double kernelize(double sim, double width, int kernelType) {
        double dist = 1.0 - sim;
        switch (kernelType) {
            case TRIANGULAR_KERNEL:
                return Math.max(1 - dist / width, 0);
            case UNIFORM_KERNEL:
                return dist < width ? 1 : 0;
            case EPANECHNIKOV_KERNEL:
                return Math.max(3.0 / 4.0 * (1 - Math.pow(dist / width, 2)), 0);
            case GAUSSIAN_KERNEL:
                return 1 / Math.sqrt(2 * Math.PI) * Math.exp(-0.5 * Math.pow(dist / width, 2));
            default:
                return Math.max(1 - dist / width, 0);
        }
    }
}
