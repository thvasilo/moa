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

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.classifiers.trees.HoeffdingTree;
import moa.classifiers.trees.RandomHoeffdingTree;

import java.util.LinkedList;
import java.util.List;
import java.util.Optional;

public class DAGLearningNode extends RandomHoeffdingTree.RandomLearningNode {

  private HoeffdingDAG tree;

  private InstanceConditionalTest splitTest;

  private boolean readyToSplit;

  private int tempLeft;
  private int tempRight;

  private ClassHistogram leftHistogram;
  private ClassHistogram rightHistogram;

  private double entropy;

  public DAGLearningNode(double[] initialClassObservations, HoeffdingDAG tree) {
    super(initialClassObservations);
    leftHistogram = new ClassHistogram();
    rightHistogram = new ClassHistogram();
    this.tree = tree;
  }

  @Override
  public AttributeSplitSuggestion[] getBestSplitSuggestions(
      SplitCriterion criterion, HoeffdingTree ht) {
    List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<AttributeSplitSuggestion>();
    double[] preSplitDist = this.observedClassDistribution.getArrayCopy();
    if (!ht.noPrePruneOption.isSet()) {
      // add null split as an option
      bestSuggestions.add(new AttributeSplitSuggestion(null,
          new double[0][], criterion.getMeritOfSplit(
          preSplitDist,
          new double[][]{preSplitDist})));
    }
    for (int i = 0; i < this.attributeObservers.size(); i++) {
      AttributeClassObserver obs = this.attributeObservers.get(i);
      if (obs != null) {
        AttributeSplitSuggestion bestSuggestion = obs.getBestEvaluatedSplitSuggestion(criterion,
            preSplitDist, i, true);
        if (bestSuggestion != null) {
          bestSuggestions.add(bestSuggestion);
        }
      }
    }
    return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
  }

  @Override
  public void learnFromInstance(Instance inst, HoeffdingTree ht) {
    // Do the learning as usual
    super.learnFromInstance(inst, ht);
    // In addition, maintain a class histogram for the left and right children
    int childBranch = splitTest.branchForInstance(inst);
    assert childBranch < 2;
    if (childBranch == 0) { // TODO: Add weight here instead of one
      leftHistogram.addOne(inst.classIndex());
    } else {
      rightHistogram.addOne(inst.classIndex());
    }
  }

  /**
   * Returns an Optional which contains the best split suggestion in the case where we should split, empty otherwise.
   * @return An optional split suggestion. If the node should be split it contains the best split suggestion,
   * otherwise it's empty.
   */
  public Optional<AttributeSplitSuggestion> checkForSplit() {

    AttributeSplitSuggestion[] bestSplitSuggestions = tree.getSplitSuggestions(this);

    Optional<AttributeSplitSuggestion> bestSplitSuggestionOptional = Optional.empty();

    if (bestSplitSuggestions.length < 2) {
      // tvas: Not sure if I want a non-empty return here, what does it mean to only have one suggestion?
//      return Optional.of(bestSplitSuggestions[bestSplitSuggestions.length - 1]);
      return bestSplitSuggestionOptional; // Returns the empty splitOptional
    } else {
      // Calculate Hoeffding bound, and check if it smaller than the merit delta between the top two suggestions
      double hoeffdingBound = HoeffdingTree.computeHoeffdingBound(
          tree.activeSplitCriterion.getRangeOfMerit(this.getObservedClassDistribution()),
          tree.splitConfidenceOption.getValue(), this.getWeightSeen());

      AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
      AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];
      if ((bestSuggestion.merit - secondBestSuggestion.merit > hoeffdingBound)
          || (hoeffdingBound < tree.tieThresholdOption.getValue())) {
        bestSplitSuggestionOptional = Optional.of(bestSplitSuggestions[bestSplitSuggestions.length - 1]);
      }

    }
    readyToSplit = bestSplitSuggestionOptional.isPresent();
    return bestSplitSuggestionOptional;
  }

  public int getTempLeft() {
    return tempLeft;
  }

  public int getTempRight() {
    return tempRight;
  }

  public ClassHistogram getLeftHistogram() {
    return leftHistogram;
  }

  public ClassHistogram getRightHistogram() {
    return rightHistogram;
  }

}
