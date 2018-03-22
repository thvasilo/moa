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

import com.bigml.histogram.SumOutOfRangeException;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.conditionaltests.NumericAttributeBinaryTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.tasks.TaskMonitor;
import moa.core.sketches.*;

import java.io.Serializable;
import java.util.ArrayList;

public class SPDTNumericClassObserver extends AbstractOptionHandler implements NumericAttributeClassObserver{

  // Contains one attribute histogram per class.
  // Describes the per class distribution for this attribute.
  private AutoExpandVector<MergeableHistogram> attHistPerClass = new AutoExpandVector<>();

  private int maxBins = 64; // TODO: Add as option
  private int numSplitPoints = 10; // TODO: Add as option

  private boolean useSPDT = true; // Set to true to use SPDT, false to use DataSketch

  private HistogramFactory histFactory = new HistogramFactory();


  private class HistogramFactory implements Serializable{
    public SPDTHistogram makeSPDT(int numBins) {
      return new SPDTHistogram(numBins);
    }

    public LDBHistogram makeSketch(int numBins) {
      return new LDBHistogram(numBins);
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

  public AutoExpandVector<MergeableHistogram> getAttHistPerClass() {
    return attHistPerClass;
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
