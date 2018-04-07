package net.librec.recommender.cf.rating;

import net.librec.common.LibrecException;
import net.librec.math.structure.*;
import net.librec.recommender.FactorizationMachineRecommender;

/**
 * Factorization Machine Recommender via Follow The Regularized Leader
 *
 * http://castellanzhang.github.io/2016/10/16/fm_ftrl_softmax
 *
 * @author Qian Shaofeng
 *
 */
public class FMFTRLRecommender extends FactorizationMachineRecommender {

    /**
     *  lambda1 is the truncated threshold
     */
    private double lambda1;

    /**
     *  lambda2 is the L2 regularization
     */
    private double lambda2;

    /**
     *  alpha and beta are used to compute learning rate.
     *  The learning rate n = alpha / ( beta + sqrt(sum(g_i^2)) )
     */
    private double alpha;
    private double beta;

    @Override
    protected void setup() throws LibrecException {
        super.setup();
        lambda1 = conf.getDouble("rec.regularization.lambda1");

        lambda2 = conf.getDouble("rec.regularization.lambda2");

        alpha = conf.getDouble("rec.learningRate.alpha");

        beta = conf.getDouble("rec.learningRate.beta");
    }

    @Override
    protected void trainModel() throws LibrecException {
        if (!isRanking){
            buildRatingModel();
        }
    }

    private void buildRatingModel() throws LibrecException {
        double zW0 = 0;
        DenseVector zW = new DenseVector(p);
        DenseMatrix zV = new DenseMatrix(p, k);
        zW.init(0);
        zV.init(0);

        double nW0 = 0;
        DenseVector nW = new DenseVector(p);
        DenseMatrix nV = new DenseMatrix(p, k);
        nW.init(0);
        nV.init(0);

        double gW0, thetaW0;

        DenseVector gW = new DenseVector(p);
        DenseVector thetaW = new DenseVector(p);

        DenseMatrix gV = new DenseMatrix(p, k);
        DenseMatrix thetaV = new DenseMatrix(p, k);

        for (int iter=0; iter < numIterations; ++iter){
            loss = 0.0;
            int userDimension = trainTensor.getUserDimension();
            int itemDimension = trainTensor.getItemDimension();
            for (TensorEntry me: trainTensor) {
                int[] entryKeys = me.keys();
                SparseVector x = tenserKeysToFeatureVector(entryKeys);
                double rate = me.get();

                // compute rating value
                double pred = predict(entryKeys[userDimension], entryKeys[itemDimension], x);

                double err = pred - rate;
                loss += err * err;

                // loss gradient, loss = 1/2 * (yhat - y)^2

                // compute w0 gradient
                gW0 = err;
                thetaW0 = 1 / alpha * (Math.sqrt(nW0 + Math.pow(gW0, 2)) - Math.sqrt(nW0));
                zW0 += gW0 - thetaW0 * w0;
                nW0 += Math.pow(gW0, 2);

                // update w0
                if (Math.abs(zW0) <= lambda1) {
                    w0 = 0;
                } else {
                    w0 = -1 / ((beta + Math.sqrt(nW0)) / alpha + lambda2) * (zW0 - sgn(zW0) * lambda1);
                }

                for(VectorEntry ve: x){
                    int i = ve.index();
                    // compute W gradient
                    double xi = ve.get();
                    gW.set(i, err * xi);
                    thetaW.set(i, 1 / alpha * (Math.sqrt(nW.get(i) + Math.pow(gW.get(i), 2)) - Math.sqrt(nW.get(i))));
                    zW.add(i, gW.get(i) - thetaW.get(i) * W.get(i));
                    nW.add(i, Math.pow(gW.get(i), 2));

                    // update W
                    if (Math.abs(zW.get(i)) <= lambda1) {
                        W.set(i, 0);
                    } else {
                        double value = -1 / ((beta + Math.sqrt(nW.get(i))) / alpha + lambda2) * (zW.get(i) - sgn(zW.get(i)) * lambda1);
                        W.set(i, value);
                    }

                    for (int f = 0; f < k; ++f) {
                        double hVlf = 0;
                        double xl =ve.get();
                        for(VectorEntry ve2: x){
                            int j = ve2.index();
                            if(j!=i){
                                hVlf += xl * V.get(j, f) * ve2.get();
                            }
                        }

                        // compute V gradient
                        double gradVlf = err * hVlf;

                        gV.set(i, f, gradVlf);
                        thetaV.set(i, f, 1 / alpha * (Math.sqrt(nV.get(i, f) + Math.pow(gV.get(i, f), 2)) - Math.sqrt(nV.get(i, f))));
                        zV.add(i, f, gV.get(i, f) - thetaV.get(i, f) * V.get(i, f));
                        nV.add(i, f, Math.pow(gV.get(i, f), 2));

                        // update V
                        if (Math.abs(zV.get(i, f)) <= lambda1) {
                            V.set(i, f, 0);
                        } else {
                            double value = -1 / ((beta + Math.sqrt(nV.get(i, f))) / alpha + lambda2) * (zV.get(i, f) - sgn(zV.get(i, f)) * lambda1);
                            V.set(i, f, value);
                        }
                    }
                }

            }

            loss *= 0.5;

            if (isConverged(iter)  && earlyStop)
                break;
        }
    }

    private int sgn(double value){
        return value > 0? 1: value==0? 0 : -1;
    }

    /**
     * This kind of prediction function cannot be applied to Factorization Machine.
     *
     * Using the predict() in FactorizationMachineRecommender class instead of this method.
     */
    @Deprecated
    protected double predict(int userIdx, int itemIdx) throws LibrecException {
        return 0;
    }
}
