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

import com.yahoo.labs.samoa.instances.Instance;

import java.util.*;

public class OoBConformalApproximate extends OoBConformalRegressor{

  private PriorityQueue<Double> sortedScores;

  private HashMap<Instance, Double> calibrationInstanceToScore;

  @Override
  protected void updateCalibrationScores() {
    assert calibrationInstanceToScore.size() <= maxCalibrationInstances;
    Collection<Double> scores = calibrationInstanceToScore.values();
    Double[] scoresArray = scores.toArray(new Double[scores.size()]);
    Arrays.sort(scoresArray);
    calibrationScores = scoresArray; // sortedScores.toArray(new Double[sortedScores.size()]);
  }

  @Override
  public void trainOnInstanceImpl(Instance inst) {

    HashMap<Integer, Double> oobTreeIndicesToPredictions = commonTraining(inst);

    if (!oobTreeIndicesToPredictions.isEmpty()) {
      double sum = 0;
      for (Double prediction : oobTreeIndicesToPredictions.values()) {
        sum += prediction;
      }
      double oobPrediction = sum / oobTreeIndicesToPredictions.size();
      double score = Math.abs(oobPrediction - inst.classValue());
      assert score >= 0;

      if (calibrationInstanceToScore.size() < maxCalibrationInstances) {

        calibrationInstanceToScore.put(inst, score);
        sortedScores.add(score);
      } else { // this way we are removing a random (?) element from the map
        Set<Instance> instanceSet = calibrationInstanceToScore.keySet();
        Instance[] instances = instanceSet.toArray(new Instance[instanceSet.size()]);

        Double removedScore = calibrationInstanceToScore.remove(instances[0]);
        calibrationInstanceToScore.put(inst, oobPrediction);
//        int prevSize = sortedScores.size();
//        if (!sortedScores.remove(removedScore)) {
//          System.out.println(Arrays.toString(sortedScores.toArray()));
//          System.out.println();
//          System.out.println(removedScore);
//          System.out.println(trainingWeightSeenByModel);
//        }
//        assert sortedScores.size() == (prevSize - 1);
//        sortedScores.add(score);
      }
    }

    updateCalibrationScores();
  }

  @Override
  public void resetLearningImpl() {
    super.resetLearningImpl();
    calibrationInstanceToScore = new HashMap<>();
    sortedScores = new PriorityQueue<>();
  }
}
