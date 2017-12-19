package moa.classifiers;
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

import java.util.concurrent.Callable;

public interface Parallel {
  void shutdownExecutor();

  class TrainingRunnable implements Runnable, Callable<Integer> {
    final private Classifier learner;
    final private Instance instance;

    public TrainingRunnable(Classifier learner, Instance instance) {
      this.learner = learner;
      this.instance = instance;
    }

    @Override
    public void run() {
      learner.trainOnInstance(this.instance);
    }

    @Override
    public Integer call() {
      run();
      return 0;
    }
  }

  class PredictionRunnable implements Runnable, Callable<double[]> {
    final private Classifier learner;
    final private Instance instance;
    private double[] votes;

    public PredictionRunnable(Classifier learner, Instance instance) {
      this.learner = learner;
      this.instance = instance;
    }

    @Override
    public void run() {
      votes = learner.getVotesForInstance(this.instance);
    }

    @Override
    public double[] call() {
      run();
      return votes;
    }
  }
}
