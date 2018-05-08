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

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.core.Example;
import moa.core.InstanceExample;

import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.DoubleStream;


public class FriedmanOneLocalExpanding extends FriedmanOneGlobalSlow {

  public IntOption thirdChangePoint = new IntOption("thirdChangePoint", 't',
      "The position of the third change point.", 750_000);
  @Override
  public Example<Instance> nextInstance() {
    numInstances++;
    double x1, x2, x3, x4, x5;
    double value;
    x1 = instanceRandom.nextDouble();
    x2 = instanceRandom.nextDouble();
    x3 = instanceRandom.nextDouble();
    x4 = instanceRandom.nextDouble();
    x5 = instanceRandom.nextDouble();
    // Original function
    value = 10 * Math.sin(Math.PI * x1 * x2) + Math.pow(20 * (x3 - 0.5), 2) + 10 * x4 + 5 * x5 + instanceRandom.nextGaussian();
    if (numInstances > firstChangePoint.getValue()) {
      // Region one
      if (x2 < 0.3 && x3 < 0.3 && x4 > 0.7 && x5 < 0.3) { // Region 1
        value = 10 * x1 * x2 + 20 * (x3 - 0.5) + 10 * x4 + 5 * x5 + instanceRandom.nextGaussian();
      } else if (x2 > 0.7 && x3 > 0.7 && x4 < 0.3 && x5 > 0.7) { // Region 2
        value = 10 * Math.cos(Math.PI * x1 * x2) + 20 * (x3 - 0.5) + Math.pow(Math.E, x4) + 5 * Math.pow(x5, 2)  + instanceRandom.nextGaussian();
      }

    }
    if (numInstances > secondChangePoint.getValue()) {
      if (x2 < 0.3 && x3 < 0.3 && x4 > 0.7) { // Region 1
        value = 10 * x1 * x2 + 20 * (x3 - 0.5) + 10 * x4 + 5 * x5 + instanceRandom.nextGaussian();
      } else if (x2 > 0.7 && x3 > 0.7 && x4 < 0.3) { // Region 2
        value = 10 * Math.cos(Math.PI * x1 * x2) + 20 * (x3 - 0.5) + Math.pow(Math.E, x4) + 5 * Math.pow(x5, 2)  + instanceRandom.nextGaussian();
      }
    }
    if (numInstances > thirdChangePoint.getValue()) {
      if (x2 < 0.3 && x3 < 0.3) { // Region 1
        value = 10 * x1 * x2 + 20 * (x3 - 0.5) + 10 * x4 + 5 * x5 + instanceRandom.nextGaussian();
      } else if (x2 > 0.7 && x3 > 0.7) { // Region 2
        value = 10 * Math.cos(Math.PI * x1 * x2) + 20 * (x3 - 0.5) + Math.pow(Math.E, x4) + 5 * Math.pow(x5, 2)  + instanceRandom.nextGaussian();
      }
    }

    InstancesHeader header = getHeader();
    Instance inst = new DenseInstance(streamHeader.numAttributes());
    inst.setValue(0, x1);
    inst.setValue(1, x2);
    inst.setValue(2, x3);
    inst.setValue(3, x4);
    inst.setValue(4, x5);
    inst.setDataset(header);
    inst.setClassValue(value);

    return new InstanceExample(inst);
  }
}
