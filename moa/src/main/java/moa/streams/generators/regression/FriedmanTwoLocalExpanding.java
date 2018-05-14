package moa.streams.generators.regression;
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

import moa.core.ObjectRepository;
import moa.tasks.TaskMonitor;

public class FriedmanTwoLocalExpanding extends FriedmanTwoGenerator {
  @Override
  protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
    super.prepareForUseImpl(monitor, repository);
    // Ugly hack to change default values
    if (firstChangePoint.getValue() == 500_000) {
      firstChangePoint.setValue(250_000);
    }
    if (secondChangePoint.getValue() == 750_000) {
      secondChangePoint.setValue(500_000);
    }
  }

  @Override
  protected double calculateValue(double x1, double x2, double x3, double x4, double x5) {
    // Original function
    double value = Math.pow(Math.pow(x1, 2) + (x2 * x3 - Math.pow(1 / (x2 * x4), 2)), 0.5) + instanceRandom.nextGaussian();

    if (numInstances > firstChangePoint.getValue()) {
      // Region one
      if (x2 < 0.3 && x3 < 0.3 && x4 > 0.7 && x5 < 0.3) { // Region 1
        value = Math.pow(Math.pow(x4, 2) + (x2 * x1 - Math.pow(1 / (x2 * x3), 2)), 0.5) + instanceRandom.nextGaussian();
      }
    }
    if (numInstances > secondChangePoint.getValue()) {
      if (x2 < 0.3 && x3 < 0.3 && x4 > 0.7) { // Region 1
        value = Math.pow(Math.pow(x4, 2) + (x2 * x1 - Math.pow(1 / (x2 * x3), 2)), 0.5) + instanceRandom.nextGaussian();
      } else if (x2 > 0.7 && x3 > 0.7 && x4 < 0.3) { // Region 2
        value = Math.pow(Math.pow(x1, 2) + (x4 * x1 - Math.pow(1 / (x1 * x3), 2)), 0.5) + instanceRandom.nextGaussian();
      }
    }
    if (numInstances > thirdChangePoint.getValue()) {
      if (x2 < 0.3 && x3 < 0.3) { // Region 1
        value = Math.pow(Math.pow(x4, 2) + (x2 * x1 - Math.pow(1 / (x2 * x3), 2)), 0.5) + instanceRandom.nextGaussian();
      } else if (x2 > 0.7 && x3 > 0.7) { // Region 2
        value = Math.pow(Math.pow(x1, 2) + (x4 * x1 - Math.pow(1 / (x1 * x3), 2)), 0.5) + instanceRandom.nextGaussian();
      }
    }

    return value;
  }
}
