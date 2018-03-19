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
import com.bigml.histogram.SumOutOfRangeException;
import com.github.javacliparser.Options;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.conditionaltests.NumericAttributeBinaryTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.AutoExpandVector;
import moa.core.DoubleVector;
import moa.core.ObjectRepository;
import moa.core.Utils;
import moa.options.AbstractOptionHandler;
import moa.options.OptionHandler;
import moa.tasks.TaskMonitor;

import java.util.ArrayList;

public class SPDTNumericClassObserver extends AbstractOptionHandler implements NumericAttributeClassObserver{

  private int maxBins = 50;

  // Contains one attribute histogram per class.
  // Describes the per class distribution for this attribute.
  private AutoExpandVector<Histogram> attHistPerClass = new AutoExpandVector<>();

  private Histogram getClassHistogram(int classVal) {
    return attHistPerClass.get(classVal);
  }

//  // TODO: Is this doing what I want it to do?
//  public void merge(SPDTNumericClassObserver other) throws MixedInsertException {
//    for (int i = 0; i < attHistPerClass.size(); i++) {
//      Histogram attHist = attHistPerClass.get(i);
//      attHist = attHist.merge(other.getClassHistogram(i));
//      attHistPerClass.set(i, attHist);
//    }
//  }

  @Override
  public void observeAttributeClass(double attVal, int classVal, double weight) {
    if (!Double.isNaN(attVal)) {
      Histogram attHist = attHistPerClass.get(classVal);
      if (attHist == null) {
        attHist = new Histogram(maxBins);
        attHistPerClass.set(classVal, attHist);
      }
      try {
        attHist.insert(attVal);
      } catch (MixedInsertException e) {
        e.printStackTrace();
      }
    }
  }

  @Override
  public double probabilityOfAttributeValueGivenClass(double attVal, int classVal) {
    return 0;
  }

  @Override
  public AttributeSplitSuggestion getBestEvaluatedSplitSuggestion(SplitCriterion criterion, double[] preSplitDist, int attIndex, boolean binaryOnly) {
    AttributeSplitSuggestion bestSuggestion = null;
    Histogram prevHist = null;
    // Merge the histograms over all classes to get the overall distribution of the attribute (at this leaf)
    for (Histogram attHist : attHistPerClass) {
      if (attHist == null) {
        continue;
      }
      if (prevHist == null) {
        prevHist = attHist;
        continue;
      }
      try {
        prevHist.merge(attHist);
      } catch (MixedInsertException e) {
        e.printStackTrace();
      }
    }

    // Get the points at which the distribution of attribute values is uniform in terms of observations per bin
    ArrayList<Double> uniformSplitPoints = prevHist.uniform(maxBins);

    // For each split point, evaluate the entropy if we were to split at that point, and select the best one
    for (Double splitPoint : uniformSplitPoints) {
      double[][] postSplitDists = new double[0][];
      try {
        postSplitDists = getClassDistsResultingFromBinarySplit(splitPoint);
      } catch (SumOutOfRangeException e) {
        e.printStackTrace();
      }
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
  public double[][] getClassDistsResultingFromBinarySplit(double splitValue) throws SumOutOfRangeException {
    DoubleVector lhsDist = new DoubleVector();
    DoubleVector rhsDist = new DoubleVector();
    for (int i = 0; i < attHistPerClass.size(); i++) {
      Histogram histogram = attHistPerClass.get(i);
      if (histogram != null) {
        double countBelow = histogram.sum(splitValue);
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
