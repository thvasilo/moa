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
  // TODO: Do I still need the null guards now that resize and reset do proper init of elements to 0?

  DefaultVector data;

  double mass = 0;

  public ClassHistogram() {
    data = new DefaultVector();
//    data.add(0, 0d);
  }

  public ClassHistogram(DefaultVector data) {
    this.data = data;
    this.mass = data.stream().mapToDouble(Double::doubleValue).sum();
  }

  public ClassHistogram(int bins) {
    data = new DefaultVector(bins);
    data.addAll((Collections.nCopies(bins == 0 ? 1 : bins, 0d)));
  }

  /**
   * Increment the class count by one for the provided class index
   */
  public void addOne(int classIndex) {
    Double value = data.get(classIndex);
    data.set(classIndex,  value == null ? 0 : value + 1);
    mass++;
  }

  /**
   * Decrement the class count by one for the provided class index
   */
  public void subOne(int classIndex) {
    Double value = data.get(classIndex);
    data.set(classIndex,  value == null ? 0 : value - 1);
    mass--;
  }

  /**
   * Increment the class count for classIndex by the given value
   */
  public void add(int classIndex, int value) {
    Double prevValue = data.get(classIndex);
    data.set(classIndex,  prevValue == null ? value : prevValue + value);
    mass += value;
  }


  /**
   * Set the value at bin classIndex to the provided value
   * @param classIndex The bin we wish to change
   * @param value The value the bin should have
   */
  public void set(int classIndex, double value) {
    mass -= data.get(classIndex) == null ? 0 : data.get(classIndex);
    data.set(classIndex, value);
    mass += value;
  }

  public void resize(int classCount) {
    data = new DefaultVector(classCount);
    data.addAll((Collections.nCopies(classCount, 0d)));
    mass = 0;
  }

  public void reset() {
    int curLength = data.size();
    data = new DefaultVector(curLength);
    data.addAll((Collections.nCopies(curLength, 0d)));
    mass = 0;
  }

  /**
   * Calculates the merged histogram of this and other, returns the merged histogram as a new object.
   * The current histogram is NOT modified!
   * @param other
   * @return A new ClassHistogram object that represents the merging of this and other.
   */
  public ClassHistogram merge(ClassHistogram other) {
    // TODO: Decide whether I want an in-place version of this as well.
    DefaultVector merged = new DefaultVector(Math.max(data.size(), other.data.size()));
    for (int i = 0; i < Math.max(data.size(), other.data.size()); i++) {
      merged.set(i, data.get(i) + other.data.get(i));
    }
    return new ClassHistogram(merged);
  }

  public double calculateEntropy() {
    if (mass < 1) return 0;

    double entropy = 0;

    for (int i = 0; i < data.size(); i++)
    {
      // Empty bins do not contribute anything
      if (data.get(i) > 0)
      {
        entropy += entropy(data.get(i)/mass);
      }
    }
    return entropy;
  }

  public double calculateCombinedEntropy(ClassHistogram other) {
    assert this.data.size() == other.data.size() : "Cannot combine histograms of different length!";

    double sum = this.mass + other.mass;

    if (sum < 1) {
      return 0;
    }

    double entropySum = 0;
    double numerator = 0;

    for (int i = 0; i < data.size(); i++) {
      // Empty bins do not contribute anything
      numerator = data.get(i) + other.data.get(i);
      if (numerator > 0) {
        entropySum += entropy(numerator / sum);
      }
    }

    return entropySum;
  }

  public double[] toArray() {
    double[] out = new double[data.size()];
    for (int i = 0; i < out.length; i++) {
      out[i] = data.get(i);
    }

    return out;
  }


  public double getMass() {
    return mass;
  }

  protected double entropy(double p) {
    return -p * Utils.log2(p);
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
  public void set(int classIndex, double value) {
    super.set(classIndex, value);
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

  public void recalculateEntropies() {
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
