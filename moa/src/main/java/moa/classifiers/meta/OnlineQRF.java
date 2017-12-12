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
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
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
 * Usage: EvaluatePrequentialRegression -e (IntervalRegressionPerformanceEvaluator -w 10000) -l (meta.OnlineQRF -t 5 -b 100 -a 90) -s (ArffFileStream -f somefile.arff)  -f 10000
 */
public class OnlineQRF  extends AbstractClassifier implements Regressor {
  private double quantileUpper;
  private double quantileLower;
  private HistogramLearner[] ensemble;
  private int ensembleSize;
  private int subspaceSize;

  private static final int FEATURES_M = 0;
  private static final int FEATURES_SQRT = 1;
  private static final int FEATURES_SQRT_INV = 2;
  private static final int FEATURES_PERCENT = 3;


  public IntOption numTrees = new IntOption("numTrees", 't', "Number of trees in the ensemble",
      5, 1, Integer.MAX_VALUE);

  // This could be a float, but I feel this is enough precision
  public IntOption confidenceLevel = new IntOption("confidenceLevel", 'a',
      "The confidence level in integer percentage points (e.g. 95 = 95% prediction interval)",
      90, 1, 100);

  public IntOption numBins = new IntOption(
      "numBins", 'b', "Number of bins to use at leaf histograms",
      100, 1, Integer.MAX_VALUE);

  public MultiChoiceOption mFeaturesModeOption = new MultiChoiceOption("mFeaturesMode", 'o',
      "Defines how m, defined by mFeaturesPerTreeSize, is interpreted. M represents the total number of features.",
      new String[]{"Specified m (integer value)", "sqrt(M)+1", "M-(sqrt(M)+1)",
          "Percentage (M * (m / 100))"},
      new String[]{"SpecifiedM", "SqrtM1", "MSqrtM1", "Percentage"}, 1);

  public IntOption mFeaturesPerTreeSizeOption = new IntOption("mFeaturesPerTreeSize", 'm',
      "Number of features allowed considered for each split. Negative values corresponds to M - m", 2, Integer.MIN_VALUE, Integer.MAX_VALUE);


  @Override
  public double[] getVotesForInstance(Instance inst) {
    // TODO: Will prolly need an option to return a single/mean value as well
    // Gather and merge all histograms
    if(this.ensemble == null)
      initEnsemble(inst);

    Histogram prevHist = null;
    for (HistogramLearner member : ensemble) {
      if (!member.learner.trainingHasStarted()) {
        return new double[] {0, 0};
      }
      if (prevHist == null) {
        prevHist = member.getPredictionHistogram(inst);
        continue;
      }
      Histogram curHist = member.getPredictionHistogram(inst);
      try {
        prevHist.merge(curHist); // tvas: Modification should happen in place, check!
      } catch (MixedInsertException e) {
        e.printStackTrace();
      }
    }
    // Get quantile from merged histograms
    assert prevHist != null;
    HashMap quantilePredictions = prevHist.percentiles(quantileLower, quantileUpper);
    if (quantilePredictions.isEmpty()) { // TODO: Why are predictions some times empty? From code seems like empty histogram, does this only happens when no data present?
      return new double[]{0, 0};
    }
    double upperPred = (double) quantilePredictions.get(quantileUpper); // tvas: Not super happy about using a double as key, see https://stackoverflow.com/q/1074781/209882. Alt iterate over map?
    double lowerPred = (double) quantilePredictions.get(quantileLower);
    return new double[]{lowerPred, upperPred};
  }

  @Override
  public void resetLearningImpl() {
    // Translate confidence to upper and lower quantiles
    double halfConfidence = (100 - confidenceLevel.getValue()) / 200.0; // We divide by two for each region (lower,upper) and by 100 to get a [0,1] double
    quantileLower = 0.0 + halfConfidence;
    quantileUpper = 1.0 - halfConfidence;
    ensembleSize = numTrees.getValue();
  }

  @Override
  public void trainOnInstanceImpl(Instance instance) {

    if(this.ensemble == null)
      initEnsemble(instance);

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

  // Mostly copied over from AdaptiveRandomForest
  protected void initEnsemble(Instance instance) {
    // Init the ensemble.
    int ensembleSize = numTrees.getValue();
    ensemble = new HistogramLearner[ensembleSize];

    subspaceSize = mFeaturesPerTreeSizeOption.getValue();

    // The size of m depends on:
    // 1) mFeaturesPerTreeSizeOption
    // 2) mFeaturesModeOption
    int numFeatures = instance.numAttributes() - 1; // Ignore class label ( -1 )

    switch(mFeaturesModeOption.getChosenIndex()) {
      case FEATURES_SQRT:
        subspaceSize = (int) Math.round(Math.sqrt(numFeatures)) + 1;
        break;
      case FEATURES_SQRT_INV:
        subspaceSize = numFeatures - (int) Math.round(Math.sqrt(numFeatures) + 1);
        break;
      case FEATURES_PERCENT:
        // If subspaceSize is negative, then first find out the actual percent, i.e., 100% - m.
        double percent = subspaceSize < 0 ? (100 + subspaceSize)/100.0 : subspaceSize / 100.0;
        subspaceSize = (int) Math.round(numFeatures * percent);
        break;
    }
    // Notice that if the selected mFeaturesModeOption was
    //  AdaptiveRandomForest.FEATURES_M then nothing is performed in the
    //  previous switch-case, still it is necessary to check (and adjusted)
    //  for when a negative value was used.

    // m is negative, use size(features) + -m
    if(subspaceSize < 0)
      subspaceSize = numFeatures + subspaceSize;
    // Other sanity checks to avoid runtime errors.
    //  m <= 0 (m can be negative if subspace was negative and
    //  abs(m) > n), then use m = 1
    if(subspaceSize <= 0)
      subspaceSize = 1;
    // m > n, then it should use n
    if(subspaceSize > numFeatures)
      subspaceSize = numFeatures;

    HistogramLearner treeLearner = new HistogramLearner();
    treeLearner.learner.resetLearning();

    for (int i = 0; i < ensembleSize; i++) {
      ensemble[i] = new HistogramLearner();
      ensemble[i].learner.prepareForUse(); // Enforce config object creation. Should be better ways to do this
      ensemble[i].learner.resetLearning();
      ensemble[i].learner.subspaceSizeOption.setValue(subspaceSize);
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
      this.learner = new FIMTQR(numBins.getValue());
    }

    public Histogram getPredictionHistogram(Instance instance) {
      return learner.getPredictionHistogram(instance);
    }
  }
}
