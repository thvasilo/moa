package moa.classifiers.meta;
/*
 * #%L
 * SAMOA
 * %%
 * Copyright (C) 2014 - 2015 Apache Software Foundation
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */

import com.bigml.histogram.Histogram;
import com.bigml.histogram.MixedInsertException;
import com.bigml.histogram.NumericTarget;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Regressor;
import moa.classifiers.trees.FIMTQR;
import moa.core.Measurement;
import moa.core.MiscUtils;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.HashMap;

/**
 * An ensemble of FIMTQR trees that are trained in the bagging manner as described in the ARF paper
 * Will need to pull out histograms from trees, in order to combine and then provide quantile prediction.
 */
public class OnlineQRF  extends AbstractClassifier implements Regressor {
  private double quantileUpper = 0.95;
  private double quantileLower = 0.05;
  private HistogramLearner[] ensemble;
  private int ensembleSize = 5;
  private long instancesSeen;

  @Override
  public double[] getVotesForInstance(Instance inst) {
    // Gather and merge all histograms
    Histogram prevHist = null;
    for (HistogramLearner member : ensemble) {
      if (!member.learner.trainingHasStarted()) {
        return new double[] {0};
      }
      if (prevHist == null) {
        prevHist = member.getPredictionHistogram(inst);
        continue;
      }
      Histogram curHist = member.getPredictionHistogram(inst);
      try {
        prevHist.merge(curHist); // Modification should happen in place, check!
      } catch (MixedInsertException e) {
        e.printStackTrace();
      }
    }
    // Get quantile from merged histograms
    assert prevHist != null;
    // Here we get both upper and lower quantiles, for now only return the upper
    // until we figure out how to use both for evaluation.
    HashMap quantilePredictions = prevHist.percentiles(quantileLower, quantileUpper);
    if (quantilePredictions.isEmpty()) { // TODO: Why are predictions some times empty? From code seems like empty histogram
      return new double[]{0};
    }
    double upperPred = (double) quantilePredictions.get(quantileUpper);
    return new double[]{upperPred};
  }

  @Override
  public void resetLearningImpl() {
    ensemble = new HistogramLearner[ensembleSize];
    for (int i = 0; i < ensembleSize; i++) {
      ensemble[i] = new HistogramLearner();
      ensemble[i].learner.prepareForUse(); // Enforce config creation. Should be better ways to do this
      ensemble[i].learner.resetLearning();
    }
  }

  @Override
  public void trainOnInstanceImpl(Instance instance) {
    ++this.instancesSeen;

    for (HistogramLearner member : ensemble) {
      // Predict and evaluate here? ARF does this, why?
//      double[] prediction = member.getVotesForInstance(instance);
      int k = MiscUtils.poisson(6.0, this.classifierRandom);
      if (k > 0) {
        Instance weightedInstance = instance.copy();
        weightedInstance.setWeight(k);
        member.learner.trainOnInstance(instance);
      }
    }
  }

  @Override
  protected Measurement[] getModelMeasurementsImpl() {
    return null;
  }

  @Override
  public void getModelDescription(StringBuilder out, int indent) {
    throw new NotImplementedException();
  }

  @Override
  public boolean isRandomizable() {
    return true;
  }

  private class HistogramLearner {
    public FIMTQR learner;

    public HistogramLearner() {
      this.learner = new FIMTQR();
    }

    public Histogram getPredictionHistogram(Instance instance) {
      return learner.getPredictionHistogram(instance);
    }
  }
}
