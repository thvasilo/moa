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

import com.bigml.histogram.Histogram;
import com.bigml.histogram.MixedInsertException;
import com.bigml.histogram.NumericTarget;
import com.bigml.histogram.SumOutOfRangeException;

import java.util.ArrayList;

/**
 * Histogram based on the Ben-Haim Streaming Parallel Decision Tree paper
 */
public class SPDTHistogram implements MergeableHistogram<SPDTHistogram> {

  private Histogram<NumericTarget> histogram;

  public SPDTHistogram(int numBins) {
    histogram = new Histogram<>(numBins);
  }

  @Override
  public void update(Double value) {
    try {
      histogram.insert(value);
    } catch (MixedInsertException e) {
      e.printStackTrace();
    }
  }

  @Override
  public SPDTHistogram merge(SPDTHistogram other) {
    try {
      // TODO: The merge will affect the internal state of the histogram, do we want that?
      histogram.merge(other.histogram);
    } catch (MixedInsertException e) {
      e.printStackTrace();
    }
    return this;
  }

  @Override
  public double[] uniform(Integer numPoints) {
    ArrayList<Double> points = histogram.uniform(numPoints);
    double[] pointsArray = new double[points.size()];
    for (int i = 0; i < points.size(); i++) {
      pointsArray[i] = points.get(i);
    }
    return pointsArray;
  }

  @Override
  public long getTotalCount() {
    return (long) histogram.getTotalCount();
  }

  @Override
  public double getSum(double p) {
    try {
      return histogram.sum(p);
    } catch (SumOutOfRangeException e) {
      e.printStackTrace();
      return 0;
    }
  }
}
