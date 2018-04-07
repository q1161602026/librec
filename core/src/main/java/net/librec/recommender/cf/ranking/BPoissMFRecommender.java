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
package net.librec.recommender.cf.ranking;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import net.librec.common.LibrecException;
import net.librec.math.algorithm.Gamma;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.DenseMatrix;
import net.librec.math.structure.MatrixEntry;
import net.librec.recommender.MatrixFactorizationRecommender;

import java.util.HashMap;
import java.util.Map;

/**
 * Prem Gopalan, et al. Scalable Recommendation with Poisson Factorization. <br>
 *
 * @author Haidong Zhang
 */

public class BPoissMFRecommender extends MatrixFactorizationRecommender {

    // The parameters of users
    private GammaDenseMatrix userTheta;

    // The parameters of items
    private GammaDenseMatrix itemBeta;

    // The parameters of multinomial;
    private Table<Integer, Integer, Map<Integer, Double>> phi;

    @Override
    protected void setup() throws LibrecException {
        super.setup();

        userTheta = new GammaDenseMatrix(numUsers, numFactors);
        itemBeta = new GammaDenseMatrix(numItems, numFactors);

        userTheta.shapePrior = conf.getDouble("rec.recommender.user.shapePrior", 0.1); // a
        userTheta.ratePrior = conf.getDouble("rec.recommender.user.ratePrior", 0.1); // b
        userTheta.init();

        itemBeta.shapePrior = conf.getDouble("rec.recommender.item.shapePrior", 0.1); // c
        itemBeta.ratePrior = conf.getDouble("rec.recommender.item.ratePrior", 0.1); // d
        itemBeta.init();


        // initialize phi
        phi = HashBasedTable.create();
        for (MatrixEntry me : trainMatrix) {
            int u = me.row();
            int i = me.column();
            phi.put(u, i, new HashMap<Integer, Double>());
        }

    }

    /**
     * update the parameters(φ) of multinomial
     * @param userTheta
     * @param user_id
     * @param itemBeta
     * @param item_id
     * @param numFactors
     * @throws LibrecException
     */
    private void updateMulti(GammaDenseMatrix userTheta, int user_id, GammaDenseMatrix itemBeta, int item_id, int numFactors) throws LibrecException {
        assert (userTheta.numColumns == itemBeta.numColumns);
        Map<Integer, Double> phi_ui = phi.get(user_id, item_id);
        double sum = 0.0d;
        double mean_log;
        double exp;
        for (int k = 0; k < numFactors; k++) {
            mean_log = 0.0d;
            mean_log += Gamma.digamma(userTheta.shape.get(user_id, k)) - Math.log(userTheta.rate.get(user_id, k));
            mean_log += Gamma.digamma(itemBeta.shape.get(item_id, k)) - Math.log(itemBeta.rate.get(item_id, k));
            exp = Math.exp(mean_log);
            phi_ui.put(k, exp);
            sum += exp;
        }
        for (int i = 0; i < numFactors; i++) {
            exp = phi_ui.get(i);
            phi_ui.put(i, exp / sum);
        }
    }

    /**
     * update the parameters(γshape,γrate) of Gamma
     * @param userTheta
     * @param itemBeta
     */
    private void updateGamma_user(GammaDenseMatrix userTheta, GammaDenseMatrix itemBeta) {
        for (int k = 0; k < numFactors; k++) {
            for (int u = 0; u < userTheta.numRows; u++) {
                userTheta.shape.set(u, k, userTheta.shapePrior);
                userTheta.rate.set(u, k, userTheta.ratePrior);
            }

            for (MatrixEntry matrixEntry : trainMatrix) {
                int user_id = matrixEntry.row();
                int item_id = matrixEntry.column();

                double val = phi.get(user_id, item_id).get(k);
                userTheta.shape.add(user_id, k, val);
                userTheta.rate.add(user_id, k, itemBeta.shape.get(item_id, k) / itemBeta.rate.get(item_id, k));
            }
        }

    }

    /**
     * update the parameters(λshape,λrate) of Gamma
     * @param userTheta
     * @param itemBeta
     */
    private void updateGamma_item(GammaDenseMatrix userTheta, GammaDenseMatrix itemBeta) {

        for (int k = 0; k < numFactors; k++) {
            for (int i = 0; i < itemBeta.numRows; i++) {
                itemBeta.shape.set(i, k, itemBeta.shapePrior);
                itemBeta.rate.set(i, k, itemBeta.ratePrior);
            }

            for (MatrixEntry matrixEntry : trainMatrix) {
                int user_id = matrixEntry.row();
                int item_id = matrixEntry.column();

                double val = phi.get(user_id, item_id).get(k);
                itemBeta.shape.add(item_id, k, val);
                itemBeta.rate.add(item_id, k, userTheta.shape.get(user_id, k) / userTheta.rate.get(user_id, k));
            }
        }

    }

    @Override
    protected void trainModel() throws LibrecException {

        for (int iter = 1; iter <= numIterations; iter++) {

            // For each user/item such that yui > 0, update the multinomial:
            for (MatrixEntry matrixEntry : trainMatrix) {
                int user_id = matrixEntry.row();
                int item_id = matrixEntry.column();
                double rating = matrixEntry.get();
                if (rating > 0) {
                    updateMulti(userTheta, user_id, itemBeta, item_id, numFactors);
                }
            }

            // For each user, update the user weight parameters:
            updateGamma_user(userTheta, itemBeta);

            // For each item, update the item weight parameters:
            updateGamma_item(userTheta, itemBeta);


        }

        userTheta.samplingParameters();
        itemBeta.samplingParameters();

    }


    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException {
        return DenseMatrix.rowMult(userTheta.value, userIdx, itemBeta.value, itemIdx);
    }

    private class GammaDenseMatrix {

        int numRows, numColumns;
        double shapePrior;
        double ratePrior;
        DenseMatrix shape;
        DenseMatrix rate;
        DenseMatrix value;


        GammaDenseMatrix(int _numRows, int _numColumns) {

            numRows = _numRows;
            numColumns = _numColumns;

            shape = new DenseMatrix(numRows, numColumns);
            rate = new DenseMatrix(numRows, numColumns);

            value = new DenseMatrix(numRows, numColumns);

        }

        void init() {
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numColumns; j++) {
                    shape.set(i, j, shapePrior + 0.01 * Randoms.uniform(0.0, 1.0));
                    rate.set(i, j, ratePrior + 0.1 * Randoms.uniform(0.0, 1.0));
                }
            }
        }

        void samplingParameters() {
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numColumns; j++) {
                    double p = Randoms.gamma(shape.get(i, j), rate.get(i, j));
                    value.set(i, j, p);
                }
            }
        }

    }

}
