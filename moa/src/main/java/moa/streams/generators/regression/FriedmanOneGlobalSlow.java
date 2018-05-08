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

import com.github.javacliparser.FloatOption;
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

/**
 * Friedman data #1 [1] with concept drift described in [2], Appendix C.
 * [1] L. Friedman, "Bagging Predictors", Machine Learning, 1996
 * [2] E. Ikonomovska "Learning model trees from evolving data streams", 2010, Data Min. Knowl. Disc.
 */
public class FriedmanOneGlobalSlow extends AbstractOptionHandler implements InstanceStream {

  public IntOption firstChangePoint = new IntOption("firstChangePoint", 'f',
      "The position of the first change point.", 250_000);

  public IntOption secondChangePoint = new IntOption("secondChangePoint", 's',
      "The position of the second change point.", 500_000);

  public IntOption instanceRandomSeedOption = new IntOption(
      "instanceRandomSeed", 'i',
      "Seed for random generation of instances.", 1);

  protected InstancesHeader streamHeader;

  protected Random instanceRandom;

  protected long numInstances;

  protected int numAttributes = 5;

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
  public Example<Instance> nextInstance() {
    numInstances++;
    double x1, x2, x3, x4, x5;
    double value;
    x1 = instanceRandom.nextDouble();
    x2 = instanceRandom.nextDouble();
    x3 = instanceRandom.nextDouble();
    x4 = instanceRandom.nextDouble();
    x5 = instanceRandom.nextDouble();
    // Global slow gradual drift from "Learning model trees from evolving data streams"
    value = 10 * Math.sin(Math.PI * x1 * x2) + Math.pow(20 * (x3 - 0.5), 2) + 10 * x4 + 5 * x5 + instanceRandom.nextGaussian();
    if (numInstances > firstChangePoint.getValue()) {
      value = 10 * Math.sin(Math.PI * x4 * x5) + Math.pow(20 * (x2 - 0.5), 2) + 10 * x1 + 5 * x3 + instanceRandom.nextGaussian();
    }
    if (numInstances > secondChangePoint.getValue()) {
      value = 10 * Math.sin(Math.PI * x2 * x5) + Math.pow(20 * (x4 - 0.5), 2) + 10 * x3 + 5 * x1 + instanceRandom.nextGaussian();
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
