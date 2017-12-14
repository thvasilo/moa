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

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import moa.classifiers.trees.FIMTQR;
import moa.core.MiscUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static moa.classifiers.meta.AdaptiveRandomForest.calculateSubspaceSize;

// There ought to be an abstract class for CP that implements the common stuff (like the arraylist for scores)
public class OoBConformalRegressor extends ConformalRegressor {

  private Classifier[] ensemble;
  private int subspaceSize;
  private int maxCalibrationInstances = 100;

  private HashMap<Instance, HashMap<Integer, Double>> calibrationInstances;

  public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
      "The number of trees.", 10, 1, Integer.MAX_VALUE);

  public MultiChoiceOption mFeaturesModeOption = new MultiChoiceOption("mFeaturesMode", 'o',
      "Defines how m, defined by mFeaturesPerTreeSize, is interpreted. M represents the total number of features.",
      new String[]{"Specified m (integer value)", "sqrt(M)+1", "M-(sqrt(M)+1)",
          "Percentage (M * (m / 100))"},
      new String[]{"SpecifiedM", "SqrtM1", "MSqrtM1", "Percentage"}, 1);

  public IntOption mFeaturesPerTreeSizeOption = new IntOption("mFeaturesPerTreeSize", 'm',
      "Number of features allowed considered for each split. Negative values corresponds to M - m", 2, Integer.MIN_VALUE, Integer.MAX_VALUE);

  public FloatOption lambdaOption = new FloatOption("lambda", 'd',
      "The lambda parameter for bagging.", 1.0, 1.0, Float.MAX_VALUE);

  @Override
  public void trainOnInstanceImpl(Instance inst) {

    HashMap<Integer, Double> oobPredictions = new HashMap<>();

    // tvas: Alternative is to have a map from instance to a tuple (predictorIndexList, predictionList)
    // That way I have access to the indices of the predictors and their corresponding predictions in a
    // perhaps easier to iterate format. Guava matrix works as well here.
    if (this.ensemble == null)
      initEnsemble(inst);
    boolean isOoB = false;
    for (int i = 0; i < ensemble.length; i++) {
      int k = MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
      if (k > 0) {
        ensemble[i].trainOnInstance(inst);
      } else {
        if (ensemble[i].trainingHasStarted()) {
          isOoB = true;
          double[] curPred =  ensemble[i].getVotesForInstance(inst);
          assert curPred.length == 1;
          oobPredictions.put(i, curPred[0]);
        }
      }
    }

    if (isOoB) {
      if (calibrationInstances.size() < maxCalibrationInstances) {
        calibrationInstances.put(inst, oobPredictions);
      } else { // this way we are removing a random (?) element from the map
        Set<Instance> instanceSet = calibrationInstances.keySet();
        Instance[] instances = instanceSet.toArray(new Instance[instanceSet.size()]);

        calibrationInstances.remove(instances[0]);
        calibrationInstances.put(inst, oobPredictions);
      }
    }

    updateCalibrationScores();
  }

  @Override
  protected void updateCalibrationScores() {
    // TODO: Optimize, not necessary to update all scores with every tree update
    double[] predictions = new double[calibrationInstances.size()];
    double[] trueValues = new double[calibrationInstances.size()];
    int i = 0; // Used to keep track of which calibration instance we are checking
    for (Map.Entry<Instance, HashMap<Integer, Double>> instancePredictionsMap : calibrationInstances.entrySet()) {
      Instance curInstance = instancePredictionsMap.getKey();
      HashMap<Integer, Double> predictorIndexPredictionMap = instancePredictionsMap.getValue();
      double sum = 0;
      for (Map.Entry<Integer, Double> predictorIndexPredictionEntry : predictorIndexPredictionMap.entrySet()) {
        sum += predictorIndexPredictionEntry.getValue();
      }
      double prediction = sum / predictorIndexPredictionMap.size();
      predictions[i] = prediction;
      trueValues[i] = curInstance.classValue();
      i++;
    }

    double[] calScores = errorFunction(predictions, trueValues);
    Arrays.sort(calScores);
    calibrationScores = calScores;
  }

  @Override
  public double[] getVotesForInstance(Instance inst) {
    if(this.ensemble == null)
      initEnsemble(inst);
    MomentAggregate curAggegate = new MomentAggregate(0, 0, 0);
    for (Classifier member : ensemble) {
      double[] curVotes = member.getVotesForInstance(inst);
      curAggegate = updateMoments(curAggegate, curVotes[0]);
    }
    curAggegate = finalizeMoments(curAggegate);
    if (calibrationInstances.size() < 10) {
      // TODO: Predictive sd is a very bad estimate, come up with something else
      // tvas: Depending on the lambda setting, it could be a while until we get 10 cal instances, careful!
      // One option: https://stats.stackexchange.com/a/255131/16052
      return new double[]{
          curAggegate.mean - Math.sqrt(curAggegate.variance),
          curAggegate.mean + Math.sqrt(curAggegate.variance)};
    }
    double interval = inverseErrorFunction(confidenceOption.getValue());
    return new double[]{curAggegate.mean - interval, curAggegate.mean + interval};
  }

  /**
   * Class used to hold results of an online mean and variance calculation
   */
  public static class MomentAggregate {
    public long count;
    public double mean;
    public double variance;

    public MomentAggregate(long count, double mean, double variance) {
      this.count = count;
      this.mean = mean;
      this.variance = variance;
    }
  }

  /**
   * Will update an existing moment aggregate
   * Source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
   * @param existingAggregate An existing aggregate of moment statistics
   * @param newValue The new value to be incorporated
   * @return An updated version of the moment aggregate. Call {finalizeMoments} on this object at the end of the stream.
   */
  public static MomentAggregate updateMoments(MomentAggregate existingAggregate, double newValue) {
    long count = existingAggregate.count;
    double mean = existingAggregate.mean;
    double variance = existingAggregate.variance;
    count++;
    double delta = newValue - mean;
    mean = mean + delta / count;
    double delta2 = newValue - mean;
    variance = variance + delta * delta2;

    return new MomentAggregate(count, mean, variance);
  }

  /**
   * Returns the finalized aggregate moments, after adjusting the sample variance
   * @param existingAggregate A moments aggregate that has all the required samples integrated.
   * @return A {@link MomentAggregate} instance with final mean and variance.
   * Variance will be NaN if less than two samples were used for the aggregate.
   */
  public static MomentAggregate finalizeMoments(MomentAggregate existingAggregate) {
    long count = existingAggregate.count;
    double mean = existingAggregate.mean;
    double variance = existingAggregate.variance;
    variance = variance / (count - 1);

    if (count < 2) {
      return new MomentAggregate(count, mean, Double.NaN);
    } else {
      return new MomentAggregate(count, mean, variance);
    }
  }

  @Override
  public void resetLearningImpl() {
    ensemble = null;
    calibrationInstances = new HashMap<>();
    maxCalibrationInstances = maxCalibrationInstancesOption.getValue();
    calibrationScores = new double[maxCalibrationInstances];
  }

  private void initEnsemble(Instance instance) {
    // Init the ensemble.
    int ensembleSize = ensembleSizeOption.getValue();
    ensemble = new FIMTQR[ensembleSize];

    subspaceSize = calculateSubspaceSize(
        mFeaturesPerTreeSizeOption.getValue(), mFeaturesModeOption.getChosenIndex(), instance);

    for (int i = 0; i < ensembleSize; i++) {
      ensemble[i] = new FIMTQR(1, subspaceSize);
      ensemble[i].prepareForUse(); // Enforce config object creation. Should be better ways to do this
      ensemble[i].resetLearning();
    }

  }
}
