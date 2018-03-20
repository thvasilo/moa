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

import java.util.Arrays;

public class ClassHistogram {
  int[] data;

  int mass = 0;

  public ClassHistogram() {
  }

  private ClassHistogram(int[] data) {
    this.data = data;
    this.mass = Arrays.stream(data).sum();
  }

  public ClassHistogram(int bins) {
    data = new int[bins];
  }

  /**
   * Increment the class count by one for the provided class index
   */
  public void addOne(int classIndex) {
    data[classIndex]++;
  }

  /**
   * Increment the class count for classIndex by the given value
   */
  public void add(int classIndex, int value) {
    data[classIndex] += value;
  }

  public void resize(int classCount) {
    data = new int[classCount];
  }

  public ClassHistogram merge(ClassHistogram other) {
    int[] merged = new int[data.length];
    for (int i = 0; i < data.length; i++) {
      merged[i] = data[i] + other.data[i];
    }
    return new ClassHistogram(merged);
  }

  public double getMass() {
    return mass;
  }
}

abstract class EntropyHistogram extends ClassHistogram {
  public EntropyHistogram(int bins) {
    super(bins);
  }

  abstract void initializeEntropies();

  abstract double entropy();
}
