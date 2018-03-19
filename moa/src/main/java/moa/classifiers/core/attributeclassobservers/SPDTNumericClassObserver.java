package moa.classifiers.core.attributeclassobservers;
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
import com.yahoo.sketches.quantiles.DoublesSketch;
import com.yahoo.sketches.quantiles.DoublesUnion;
import com.yahoo.sketches.quantiles.UpdateDoublesSketch;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.conditionaltests.NumericAttributeBinaryTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.tasks.TaskMonitor;

import java.io.Serializable;
import java.util.ArrayList;

public class SPDTNumericClassObserver extends AbstractOptionHandler implements NumericAttributeClassObserver{

  // Contains one attribute histogram per class.
  // Describes the per class distribution for this attribute.
  private AutoExpandVector<MergeableHistogram> attHistPerClass = new AutoExpandVector<>();

  private int maxBins = 64; // TODO: Add as option
  private int numSplitPoints = 10; // TODO: Add as option

  private boolean useSPDT = false;

  private HistogramFactory histFactory = new HistogramFactory();

  private interface MergeableHistogram {

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

  private class HistogramFactory implements Serializable{
    public SPDTHistogram makeSPDT(int numBins) {
      return new SPDTHistogram(numBins);
    }

    public SketchHistogram makeSketch(int numBins) {
      return new SketchHistogram(numBins);
    }
  }

  private class SketchHistogram implements MergeableHistogram {

    private UpdateDoublesSketch histogram;

    @Override
    public void update(Double value) {

    }

    public SketchHistogram(int numBins) {
      histogram = DoublesSketch.builder().setK(numBins).build();
    }

    @Override
    public MergeableHistogram merge(MergeableHistogram other) {
      if (other instanceof SketchHistogram) {
        SketchHistogram otherSketch = (SketchHistogram) other;

        DoublesUnion union = DoublesUnion.builder().build();
        union.update(this.histogram);
        union.update(otherSketch.histogram);

        this.histogram = union.getResult();
      }
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



  private class SPDTHistogram implements MergeableHistogram {

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
    public MergeableHistogram merge(MergeableHistogram other) {
      if (other instanceof SPDTHistogram) {
        SPDTHistogram otherSPDT = (SPDTHistogram) other;
        try {
          // TODO: The merge will affect the internal state of the histogram, do we want that?
          histogram.merge(otherSPDT.histogram);
        } catch (MixedInsertException e) {
          e.printStackTrace();
        }
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


  @Override
  public void observeAttributeClass(double attVal, int classVal, double weight) {
    if (!Double.isNaN(attVal)) {
      MergeableHistogram attHist = attHistPerClass.get(classVal);
      if (attHist == null) {

        attHist = useSPDT ? histFactory.makeSPDT(maxBins) : histFactory.makeSketch(maxBins);
        attHistPerClass.set(classVal, attHist);
      }
      attHist.update(attVal);
    }
  }

  @Override
  public double probabilityOfAttributeValueGivenClass(double attVal, int classVal) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public AttributeSplitSuggestion getBestEvaluatedSplitSuggestion(SplitCriterion criterion, double[] preSplitDist, int attIndex, boolean binaryOnly) {
    AttributeSplitSuggestion bestSuggestion = null;
    MergeableHistogram prevHist = null;
    // Merge the histograms over all classes to get the overall distribution of the attribute (at this leaf)
    for (MergeableHistogram attHist : attHistPerClass) {
      if (attHist == null) {
        continue;
      }
      if (prevHist == null) {
        prevHist = attHist;
        continue;
      }
      prevHist.merge(attHist);
    }

    if (prevHist == null || prevHist.getTotalCount() == 0) {
      return null;
    }
    // Get the points at which the distribution of attribute values is uniform in terms of observations per bin
    double[] uniformSplitPoints = prevHist.uniform(numSplitPoints);

    // For each split point, evaluate the entropy if we were to split at that point, and select the best one
    for (Double splitPoint : uniformSplitPoints) {
      double[][] postSplitDists;
      postSplitDists = getClassDistsResultingFromBinarySplit(splitPoint);
      double merit = criterion.getMeritOfSplit(preSplitDist,
          postSplitDists);
      if ((bestSuggestion == null) || (merit > bestSuggestion.merit)) {
        bestSuggestion = new AttributeSplitSuggestion(
            new NumericAttributeBinaryTest(attIndex,
                splitPoint, true), postSplitDists, merit);
      }
    }

    return bestSuggestion;
  }


  /**
   * Returns the class distribution resulting by splitting the stored per class histograms at the provided split value,
   * for the left and right split.
   * @param splitValue The attribute value to split on.
   * @return A 2D vector (matrix). The first index ("rows") corresponds to the left and right hand side class distribution
   * respectively, i.e. return[0] is lhs class distribution, return[1] is rhs.
   * The second index ("columns") is the distribution (counts) for each class for the given split side.
   * So return[0][0] is the count of examples on the left side of the split for class 0.
   * @throws SumOutOfRangeException
   */
  public double[][] getClassDistsResultingFromBinarySplit(double splitValue) {
    DoubleVector lhsDist = new DoubleVector();
    DoubleVector rhsDist = new DoubleVector();
    for (int i = 0; i < attHistPerClass.size(); i++) {
      MergeableHistogram histogram = attHistPerClass.get(i);
      if (histogram != null) {
        double countBelow = histogram.getSum(splitValue);
        // TODO: For sketch, get CDF then multiply with number of points.
        lhsDist.addToValue(i, countBelow);
        rhsDist.addToValue(i, histogram.getTotalCount() - countBelow);
      }
    }
    return new double[][]{lhsDist.getArrayRef(), rhsDist.getArrayRef()};
  }

  @Override
  public void observeAttributeTarget(double attVal, double target) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
    // TODO Auto-generated method stub
  }

  @Override
  public void getDescription(StringBuilder sb, int indent) {
    // TODO Auto-generated method stub
  }
}
