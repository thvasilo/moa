package moa.core.sketches;
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

public interface MergeableHistogram {

  /**
   * Insert a value to the histogram.
   * @param value
   */
  void update(Double value);

  /**
   * Merge the current histogram with another. Modifies the state of the internal histogram.
   * @param other A Histogram object, must be of the same class.
   * @return Returns the histogram object after modification.
   */
  MergeableHistogram merge(MergeableHistogram other);

  /**
   * Will derive a list of split points at which the histogram bins have approximately equal number of data points.
   * @param numPoints The desired number of uniform bins.
   * @return
   */
  double[] uniform(Integer numPoints);

  /**
   * Get the total number of points in the histogram.
   * @return
   */
  long getTotalCount();

  /**
   * Get the number of data points which are less than or equal to the provided value.
   * @param p The cutoff point for the cumulative sum
   * @return The number of elements in the histogram less than or equal to p
   */
  double getSum(double p);
}
