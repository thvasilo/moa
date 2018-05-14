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

/**
 * Friedman data #1 [1] with concept drift described in [2], Appendix C.
 * [1] L. Friedman, "Bagging Predictors", Machine Learning, 1996
 * [2] E. Ikonomovska "Learning model trees from evolving data streams", 2010, Data Min. Knowl. Disc.
 */
public class FriedmanOneGlobalSlow extends FriedmanOneGenerator {

  @Override
  protected double calculateValue(double x1, double x2, double x3, double x4, double x5) {
    double value;
    // Global slow gradual drift from "Learning model trees from evolving data streams"
    value = 10 * Math.sin(Math.PI * x1 * x2) + Math.pow(20 * (x3 - 0.5), 2) + 10 * x4 + 5 * x5 + instanceRandom.nextGaussian();
    if (numInstances > firstChangePoint.getValue()) {
      long instancesSinceChange = numInstances - firstChangePoint.getValue();
      if (instancesSinceChange / driftLength.getValue() > instanceRandom.nextDouble()) {
        value = 10 * Math.sin(Math.PI * x4 * x5) + Math.pow(20 * (x2 - 0.5), 2) + 10 * x1 + 5 * x3 + instanceRandom.nextGaussian();
      }
    }
    if (numInstances > secondChangePoint.getValue()) {
      long instancesSinceChange = numInstances - secondChangePoint.getValue();
      if (instancesSinceChange / driftLength.getValue() > instanceRandom.nextDouble()) {
        value = 10 * Math.sin(Math.PI * x2 * x5) + Math.pow(20 * (x4 - 0.5), 2) + 10 * x3 + 5 * x1 + instanceRandom.nextGaussian();
      }
    }

    return value;
  }
}
