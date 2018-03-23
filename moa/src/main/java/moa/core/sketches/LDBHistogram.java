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

import com.yahoo.sketches.quantiles.DoublesSketch;
import com.yahoo.sketches.quantiles.DoublesUnion;
import com.yahoo.sketches.quantiles.UpdateDoublesSketch;

/**
 * Histogram based on the  PK Agarwal "Mergeable Summaries" paper, using Yahoo DataSketches implementation
 */
public class LDBHistogram implements MergeableHistogram<LDBHistogram> {

  private UpdateDoublesSketch histogram;

  @Override
  public void update(Double value) {
    histogram.update(value);
  }

  public LDBHistogram(int numBins) {
    histogram = DoublesSketch.builder().setK(numBins).build();
  }

  private LDBHistogram() {
    histogram = DoublesSketch.builder().build();
  }

  @Override
  public LDBHistogram merge(LDBHistogram other) {
    DoublesUnion union = DoublesUnion.builder().build();
    union.update(this.histogram);
    union.update(other.histogram);
    this.histogram = union.getResult();
    return this;
  }

  @Override
  public double[] uniform(Integer numPoints) {
    return histogram.getQuantiles(numPoints);
  }

  @Override
  public long getTotalCount() {
    return histogram.getN();
  }

  @Override
  public double getSum(double p) {
    double[] cdfPoints = histogram.getCDF(new double[]{p});
    return cdfPoints[0] * getTotalCount();
  }
}
