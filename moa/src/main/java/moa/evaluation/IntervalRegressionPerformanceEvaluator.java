package moa.evaluation;
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

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Prediction;
import moa.core.Example;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.tasks.TaskMonitor;
import moa.evaluation.WindowRegressionPerformanceEvaluator.Estimator;


public class IntervalRegressionPerformanceEvaluator extends AbstractOptionHandler
    implements RegressionPerformanceEvaluator {

  private double totalWeightObserved = 0;

  public IntOption widthOption = new IntOption("width",
      'w', "Size of Window", 1000);

  private Estimator errorRate;
  private Estimator windowWeightObserved;

  @Override
  public void reset() {
    errorRate = new Estimator(widthOption.getValue());
    windowWeightObserved = new Estimator(widthOption.getValue());
  }

  @Override
  public void addResult(Example<Instance> testInst, Prediction prediction) {
    double votes[];
    if(prediction==null)
      votes = new double[]{0,0}; // tvas: Not sure about the validity of this approach
    else
      votes=prediction.getVotes();
    addResult(testInst, votes);
  }

  @Override
  public void addResult(Example<Instance> example, double[] prediction) {
    Instance inst = example.getData();
    double trueValue = inst.classValue();
    if (totalWeightObserved == 0) {
      reset(); // Original has number of classes as arg here, why?
    }
    totalWeightObserved += 1.0;
    windowWeightObserved.add(1.0); // tvas: Don't want to mess with weights for now
    assert prediction.length == 2;
    assert prediction[0] <= prediction[1]; // TODO: Remove assertions if doing timing tests
    // Check if the true value is within the given interval
    boolean withinInterval = (trueValue >= prediction[0]) && (trueValue <= prediction[1]);
    // Add one to the error rate only if it's not
    errorRate.add(withinInterval ? 0 : 1);
  }

  @Override
  public Measurement[] getPerformanceMeasurements() {
    return new Measurement[]{
        new Measurement("classified instances",
            totalWeightObserved),
        new Measurement("mean error rate",
            meanErrorRate())};
  }

  private double meanErrorRate() {
    return errorRate.total()/ windowWeightObserved.total();
  }

  @Override
  protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {

  }

  @Override
  public void getDescription(StringBuilder sb, int indent) {

  }
}
