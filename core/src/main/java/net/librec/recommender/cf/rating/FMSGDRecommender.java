/**
 * Copyright (C) 2016 LibRec
 * <p>
 * This file is part of LibRec.
 * LibRec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * LibRec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with LibRec. If not, see <http://www.gnu.org/licenses/>.
 */

package net.librec.recommender.cf.rating;

import net.librec.annotation.ModelData;
import net.librec.common.LibrecException;
import net.librec.math.structure.SparseVector;
import net.librec.math.structure.TensorEntry;
import net.librec.math.structure.VectorEntry;
import net.librec.recommender.FactorizationMachineRecommender;


/**
 * Stochastic Gradient Descent with Square Loss
 * Rendle, Steffen, "Factorization Machines", Proceedings of the 10th IEEE International Conference on Data Mining, 2010
 * Rendle, Steffen, "Factorization Machines with libFM", ACM Transactions on Intelligent Systems and Technology, 2012
 *
 * @author Jiaxi Tang and Ma Chen
 */

@ModelData({"isRanking", "fmsgd", "W", "V", "W0", "k"})
public class FMSGDRecommender extends FactorizationMachineRecommender {
    /**
     * learning rate of stochastic gradient descent
     */
    private double learnRate;

    @Override
    protected void setup() throws LibrecException {
        super.setup();
        learnRate = conf.getDouble("rec.iterator.learnRate");
    }

    @Override
    protected void trainModel() throws LibrecException {
        if (!isRanking) {
            buildRatingModel();
        }
    }

    private void buildRatingModel() throws LibrecException {
        int userDimension = trainTensor.getUserDimension();
        int itemDimension = trainTensor.getItemDimension();

        for (int iter = 0; iter < numIterations; iter++) {
            lastLoss = loss;
            loss = 0.0;
            for (TensorEntry me : trainTensor) {
                int[] entryKeys = me.keys();
                SparseVector x = tenserKeysToFeatureVector(entryKeys);

                double rate = me.get();
                double pred = predict(entryKeys[userDimension], entryKeys[itemDimension], x);

                double err = pred - rate;
                loss += err * err;

                // global bias
                loss += regW0 * w0 * w0;

                double hW0 = 1;
                double gradW0 = err * hW0 + regW0 * w0;

                // update w0
                w0 += -learnRate * gradW0;

                // 1-way interactions
                for(VectorEntry ve: x){
                    int i = ve.index();
                    double oldWi = W.get(i);
                    double xi = ve.get();
                    double gradWl = err * xi + regW * oldWi;
                    W.add(i, -learnRate * gradWl);

                    loss += regW * oldWi * oldWi;

                    // 2-way interactions
                    for (int f = 0; f < k; f++) {
                        double oldVlf = V.get(i, f);
                        double hVif = 0;
                        for(VectorEntry ve2: x){
                            int j = ve2.index();
                            double xj = ve2.get();

                            if(j!=i){
                                hVif += xi * V.get(j, f) * xj;
                            }
                        }

                        double gradVlf = err * hVif + regF * oldVlf;
                        V.add(i, f, -learnRate * gradVlf);
                        loss += regF * oldVlf * oldVlf;
                    }
                }

            }

            loss *= 0.5;

            if (isConverged(iter)  && earlyStop)
                break;
        }
    }


    /**
     * This kind of prediction function cannot be applied to Factorization Machine.
     *
     * Using the predict() in FactorizationMachineRecommender class instead of this method.
     */
    @Deprecated
    protected double predict(int userIdx, int itemIdx) throws LibrecException {
        return 0.0;
    }
}
