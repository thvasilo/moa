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

public class FriedmanThreeGlobalSlow extends FriedmanTwoGenerator{
  @Override
  protected double calculateValue(double x1, double x2, double x3, double x4, double x5) {
    // Global slow gradual drift from "Learning model trees from evolving data streams"
    double value = Math.atan((x2 * x3 - 1/(x2 *x4)) / x1) + instanceRandom.nextGaussian();
    if (numInstances > firstChangePoint.getValue()) {
      long instancesSinceChange = numInstances - firstChangePoint.getValue();
      if ((double) instancesSinceChange / driftLength.getValue() > instanceRandom.nextDouble()) {
        value = Math.atan( (x1 * x4 - 1/(x2 *x3)) / x2) + instanceRandom.nextGaussian();
      }
    }
    if (numInstances > secondChangePoint.getValue()) {
      long instancesSinceChange = numInstances - secondChangePoint.getValue();
      if ((double) instancesSinceChange / driftLength.getValue() > instanceRandom.nextDouble()) {
        value = Math.atan( (x2 * x4 - 1/(x2 *x1)) / x1) + instanceRandom.nextGaussian();
      }
    }
    return value;
  }
}
