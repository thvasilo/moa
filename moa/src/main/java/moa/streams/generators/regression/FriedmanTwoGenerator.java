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

import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.core.Example;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.tasks.TaskMonitor;

abstract public class FriedmanTwoGenerator extends FriedmanOneGenerator {

  int iterationLimit = 1000;


  @Override
  public Example<Instance> nextInstance() {
    numInstances++;

    double x1 = 0, x2 = 0, x3 = 0, x4 = 0;
    double value = Double.NaN;

    int iterations = 0;
    while (Double.isNaN(value)) {
      iterations++;
      // Hack: Guard against infinite loop
      if (iterations > iterationLimit) {
        System.out.println("Exiting, could not get a non-NaN value within iteration limit!");
        System.exit(1);
      }
      x1 = instanceRandom.nextDouble() * 100;
      assert x1 >= 0 && x1 <= 100 : x1;
      x2 = instanceRandom.nextDouble() * 520 * Math.PI;
      x2 += 40 * Math.PI;
      assert x2 >= 40 * Math.PI && x2 <= 560 * Math.PI : x2;
      x3 = instanceRandom.nextDouble();
      x4 = instanceRandom.nextDouble() * 10;
      x4 += 1;
      assert x4 >= 1 && x4 <= 11 : x4;

      value = calculateValue(x1, x2, x3, x4, -1); // Hack, abusing function contract
    }

    assert !Double.isNaN(value);
    InstancesHeader header = getHeader();
    Instance inst = new DenseInstance(streamHeader.numAttributes());
    inst.setValue(0, x1);
    inst.setValue(1, x2);
    inst.setValue(2, x3);
    inst.setValue(3, x4);
    inst.setDataset(header);
    inst.setClassValue(value);

    return new InstanceExample(inst);
  }

  @Override
  protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
    numInstances = 4;
    super.prepareForUseImpl(monitor, repository);
  }
}
