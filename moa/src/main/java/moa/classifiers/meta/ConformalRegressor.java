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
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.Regressor;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.streams.ArffFileStream;

import java.util.ArrayList;
import java.lang.Math;
import java.util.Arrays;

public class ConformalRegressor extends AbstractClassifier implements Regressor{
  public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
      "Classifier to train.", Classifier.class, "trees.FIMTDD");

  public StringOption calibrationDataset = new StringOption("calibrationData", 'c',
      "Filepath to the arff file to be used as a calibration dataset", "");

  public FloatOption confidence = new FloatOption("confidence", 'a',
      "Prediction confidence level. Set to 0.9 for a 90% prediction interval.", 0.9);

  public IntOption maxCalibrationInstances = new IntOption("maxCalibrationInstances", 'm',
      "Maximum number of instances to use for calibration", 1000, 10, Integer.MAX_VALUE);

  private Classifier model;

  private ArrayList<Instance> calibrationSet;

  private double[] calibrationScores;

  private void readCalibrationSet() {
    ArffFileStream stream = new ArffFileStream(calibrationDataset.getValue(), -1);
    int i = 0;
    while (stream.hasMoreInstances() && i < maxCalibrationInstances.getValue()) {
      calibrationSet.add(stream.nextInstance().instance);
      i++;
    }
  }

  private double[] reverseArray(double[] validData) {
    for(int i = 0; i < validData.length / 2; i++)
    {
      double temp = validData[i];
      validData[i] = validData[validData.length - i - 1];
      validData[validData.length - i - 1] = temp;
    }

    return validData;
  }

  private void updateCalibrationSet() {
    // tvas: To be used with OoB predictors
  }

  // TODO: Take calibration set input

  private double[] errorFunction(double[] predictions, double[] trueValues) {
    // Implementing the absolute error function for now, this could be generalized
    double[] scores = new double[trueValues.length];
    // TODO: Should be done in jBLAS or something
    for (int i = 0; i < trueValues.length; i++) {
      scores[i] = Math.abs(predictions[i] - trueValues[i]);
    }

    return scores;
  }

  /**
   * Incerse of non-conformity function, i.e. calculates prediction interval (PI).
   * @param significance Interval confidence. Example: If we want 90% PIs this should be 0.1
   * @return The value to add/subtract on each side of the prediction to get the PI
   */
  private double inverseErrorFunction(double significance) {
    // tvas: Maybe this should be a class parameter to avoid re-calculation?
    // tvas: We assume the calibration scores are up-to-date and sorted
//    double[] calibrationScores = reverseArray(this.calibrationScores); // TODO: Not necessary if I just pass confidence as significance?
    int border = (int) Math.floor(significance * (calibrationScores.length + 1)) - 1;
    border = Math.min(Math.max(border, 0), calibrationScores.length - 1);
    return calibrationScores[border];
  }

  private void updateCalibrationScores() {
    double[] predictions = new double[calibrationSet.size()];
    double[] trueValues = new double[calibrationSet.size()];
    for (int i = 0; i < predictions.length; i++) {
      Instance calInstance = calibrationSet.get(i);
      trueValues[i] = calInstance.classValue();
      double[] prediction = model.getVotesForInstance(calInstance);
      assert prediction.length == 1;
      predictions[i] = prediction[0];
    }
    double[] calScores = errorFunction(predictions, trueValues);
    Arrays.sort(calScores); // tvas: mah performance!
    calibrationScores = calScores;
  }

  @Override
  public double[] getVotesForInstance(Instance inst) {
    if (!model.trainingHasStarted()) {
      return new double[] {0, 0};
    }
    double[] modelPredictionArray = model.getVotesForInstance(inst);
    assert modelPredictionArray.length == 1;
    double modelPrediction = modelPredictionArray[0];
//    double interval = inverseErrorFunction(1.0 - confidence.getValue());
    double interval = inverseErrorFunction(confidence.getValue());
    // TODO: Normalized prediction intervals
    return new double[]{modelPrediction - interval, modelPrediction + interval};
  }

  @Override
  public void resetLearningImpl() {
    model = (Classifier) getPreparedClassOption(baseLearnerOption);
    assert model instanceof Regressor; // Will this work?
    calibrationSet = new ArrayList<>();
    model.resetLearning();
    readCalibrationSet();
  }

  @Override
  public void trainOnInstanceImpl(Instance inst) {
    model.trainOnInstance(inst);
    updateCalibrationSet();
    updateCalibrationScores(); // tvas: This could happen async as well, just needs to complete before next prediction
  }

  @Override
  protected Measurement[] getModelMeasurementsImpl() {
    return new Measurement[0];
  }

  @Override
  public void getModelDescription(StringBuilder out, int indent) {

  }

  @Override
  public boolean isRandomizable() {
    return false;
  }



}
