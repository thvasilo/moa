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

public class FriedmanTwoGlobalAbruptReoccurring extends FriedmanTwoGenerator{
  @Override
  protected double calculateValue(double x1, double x2, double x3, double x4, double x5) {
    // Original function
    double value = Math.pow(Math.pow(x1, 2) + (x2 * x3 - Math.pow(1 / (x2 * x4), 2)), 0.5) + instanceRandom.nextGaussian();
    if (numInstances > firstChangePoint.getValue()) {
      // Abrupt drift
      value = Math.pow(Math.pow(x4, 2) + (x2 * x1 - Math.pow(1 / (x2 * x3), 2)), 0.5) + instanceRandom.nextGaussian();
    }
    if (numInstances > secondChangePoint.getValue()) {
      // Return to original
      value = Math.pow(Math.pow(x1, 2) + (x2 * x3 - Math.pow(1 / (x2 * x4), 2)), 0.5) + instanceRandom.nextGaussian();
    }

    return value;
  }
}
