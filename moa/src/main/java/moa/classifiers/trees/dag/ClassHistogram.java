package moa.classifiers.trees.dag;
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

import moa.core.AutoExpandVector;
import moa.core.Utils;

import java.util.Collections;

public class ClassHistogram {
  // TODO: Add null pointer guards to all operations
  AutoExpandVector<Integer> data;

  int mass = 0;

  public ClassHistogram() {
    data = new AutoExpandVector<>();
    data.add(0, 0);
  }

  public ClassHistogram(AutoExpandVector<Integer> data) {
    this.data = data;
    this.mass = data.stream().mapToInt(Integer::intValue).sum();
  }

  public ClassHistogram(int bins) {
    data = new AutoExpandVector<>(bins);
    data.addAll((Collections.nCopies(bins == 0 ? 1 : bins, 0)));
  }

  /**
   * Increment the class count by one for the provided class index
   */
  public void addOne(int classIndex) {
    Integer value = data.get(classIndex);
    data.set(classIndex,  value == null ? 0 : value + 1);
    mass++;
  }

  /**
   * Decrement the class count by one for the provided class index
   */
  public void subOne(int classIndex) {
    Integer value = data.get(classIndex);
    data.set(classIndex,  value == null ? 0 : value - 1);
    mass--;
  }

  /**
   * Increment the class count for classIndex by the given value
   */
  public void add(int classIndex, int value) {
    Integer prevValue = data.get(classIndex);
    data.set(classIndex,  prevValue == null ? value : prevValue + value);
    mass += value;
  }


  /**
   * Set the value at bin classIndex to the provided value
   * @param classIndex The bin we wish to change
   * @param value The value the bin should have
   */
  public void set(int classIndex, int value) {
    mass -= data.get(classIndex) == null ? 0 : data.get(classIndex);
    data.set(classIndex, value);
    mass += value;
  }

  public void resize(int classCount) {
    data = new AutoExpandVector<>(classCount);
    mass = 0;
  }

  public void reset() {
    data = new AutoExpandVector<>(data.size());
    mass = 0;
  }

  /**
   * Calculates the merged histogram of this and other, returns a new object.
   * @param other
   * @return
   */
  public ClassHistogram merge(ClassHistogram other) {
    AutoExpandVector<Integer> merged = new AutoExpandVector<>(data.size());
    for (int i = 0; i < data.size(); i++) {
      merged.set(i, data.get(i) + other.data.get(i));
    }
    return new ClassHistogram(merged);
  }

  public double getMass() {
    return mass;
  }
}

class EntropyHistogram extends ClassHistogram {
  private AutoExpandVector<Double> entropies;
  private double totalEntropy;

  public EntropyHistogram(int bins) {
    super(bins);
    entropies = new AutoExpandVector<>(bins);
  }

  // TODO: add value method that keeps entropy up-to-date

  public EntropyHistogram(ClassHistogram classHistogram) {
    data = classHistogram.data;
    mass = classHistogram.mass;
    entropies = new AutoExpandVector<>(data.size());
    recalculateEntropies();
  }

  @Override
  public void set(int classIndex, int value) {
    super.set(classIndex, value);
    recalculateEntropies();
  }

  @Override
  public void addOne(int classIndex) {
    totalEntropy += entropy(getMass());
    mass++;
    totalEntropy += -entropy(getMass());

    data.set(classIndex, data.get(classIndex) + 1);
    totalEntropy -= entropies.get(classIndex);
    entropies.set(classIndex, entropy(data.get(classIndex)));
    totalEntropy += entropies.get(classIndex);
  }

  @Override
  public void subOne(int classIndex) {
    totalEntropy += entropy(getMass());
    mass--;
    totalEntropy += -entropy(getMass());

    data.set(classIndex, data.get(classIndex) - 1);
    totalEntropy -= entropies.get(classIndex);
    entropies.set(classIndex, data.get(classIndex) < 1 ? 0 : entropy(data.get(classIndex)));
    totalEntropy += entropies.get(classIndex);
  }

  private double entropy(double p) {
    return -p * Utils.log2(p);
  }

  private void recalculateEntropies() {
    if (getMass() < 1)
    {
      totalEntropy = 0;
      for (int i = 0; i < data.size(); i++)
      {
        entropies.set(i, 0d);
      }
    }
    else
    {
      totalEntropy = -entropy(getMass());
      for (int i = 0; i < data.size(); i++)
      {
        if (data.get(i) == 0) continue;

        entropies.set(i, entropy(data.get(i)));

        totalEntropy += entropies.get(i);
      }
    }
  }

  public double entropy() {
    return totalEntropy;
  }

  public ClassHistogram toClassHistogram() {
    return new ClassHistogram(data);
  }
}
