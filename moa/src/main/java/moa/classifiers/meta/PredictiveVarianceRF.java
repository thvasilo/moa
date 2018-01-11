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
import moa.core.MiscUtils;

import java.util.ArrayList;
import java.util.Collection;

public class PredictiveVarianceRF extends OoBConformalRegressor {

  @Override
  public void trainOnInstanceImpl(Instance inst) {
    if (this.ensemble == null)
      initEnsemble(inst);

    Collection<TrainingRunnable> inBag = new ArrayList<>();
    for (int i = 0; i < ensemble.length; i++) {
      int k = MiscUtils.poisson(this.lambdaOption.getValue(), this.classifierRandom);
      if (k > 0) {
        Instance weightedInstance = inst.copy();
        weightedInstance.setWeight(k);
        if(this.executor != null) {
          TrainingRunnable trainer = new TrainingRunnable(ensemble[i],
              weightedInstance);
          inBag.add(trainer);
        }
        else {
          ensemble[i].trainOnInstance(weightedInstance);
        }
      }
    }

    if(executor != null) {
      try {
        executor.invokeAll(inBag);
      } catch (InterruptedException ex) {
        throw new RuntimeException("Could not call invokeAll() on training threads.");
      }
    }

  }

  @Override
  public double[] getVotesForInstance(Instance inst) {
    MomentAggregate curAggegate = getMoments(inst);
    // TODO: Assume Gaussian distribution with given mean/variance and return quantiles
    return calculateGaussianInterval(curAggegate.mean, Math.sqrt(curAggegate.variance));

  }
}
