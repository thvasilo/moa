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
import com.yahoo.labs.samoa.instances.*;
import moa.core.Example;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;

import java.util.ArrayList;
import java.util.Random;

abstract public class FriedmanOneGenerator extends AbstractOptionHandler implements InstanceStream {
  public IntOption firstChangePoint = new IntOption("firstChangePoint", 'f',
      "The position of the first change point.", 500_000);

  public IntOption secondChangePoint = new IntOption("secondChangePoint", 's',
      "The position of the second change point.", 750_000);

  public IntOption thirdChangePoint = new IntOption("thirdChangePoint", 't',
      "The position of the third change point.", 750_000);

  public IntOption driftLength = new IntOption("driftLength", 'd',
      "The number of instances it takes to switch from one concept to the next.", 100_000);

  public IntOption instanceRandomSeedOption = new IntOption(
      "instanceRandomSeed", 'i',
      "Seed for random generation of instances.", 1);

  protected InstancesHeader streamHeader;

  protected Random instanceRandom;

  protected long numInstances;

  protected int numAttributes = 10;

  @Override
  public Example<Instance> nextInstance() {
    numInstances++;
    double x1, x2, x3, x4, x5;
    x1 = instanceRandom.nextDouble();
    x2 = instanceRandom.nextDouble();
    x3 = instanceRandom.nextDouble();
    x4 = instanceRandom.nextDouble();
    x5 = instanceRandom.nextDouble();

    double value = calculateValue(x1, x2, x3, x4, x5);

    InstancesHeader header = getHeader();
    Instance inst = new DenseInstance(streamHeader.numAttributes());
    inst.setValue(0, x1);
    inst.setValue(1, x2);
    inst.setValue(2, x3);
    inst.setValue(3, x4);
    inst.setValue(4, x5);
    // Generate irrelevant attributes
    for (int i = 5; i < 10; i++) {
      inst.setValue(i, instanceRandom.nextDouble());
    }
    inst.setDataset(header);
    inst.setClassValue(value);

    return new InstanceExample(inst);
  }

  abstract protected double calculateValue(double x1, double x2, double x3, double x4, double x5);

  @Override
  protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
    numInstances = 0;

    ArrayList<Attribute> attributes = new ArrayList();

    for (int i = 0; i < numAttributes; i++) {
      attributes.add(new Attribute("x" + (i + 1)));
    }
    Attribute classAtt = new Attribute("value");
    attributes.add(classAtt);


    this.streamHeader = new InstancesHeader(new Instances(
        getCLICreationString(InstanceStream.class), attributes, 0));
    this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
    restart();
  }

  @Override
  public InstancesHeader getHeader() {
    return streamHeader;
  }

  @Override
  public long estimatedRemainingInstances() {
    return -1;
  }

  @Override
  public boolean hasMoreInstances() {
    return true;
  }

  @Override
  public boolean isRestartable() {
    return true;
  }

  @Override
  public void restart() {
    this.instanceRandom = new Random(
        this.instanceRandomSeedOption.getValue());
  }

  @Override
  public void getDescription(StringBuilder sb, int indent) {

  }
}
