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
import moa.classifiers.core.attributeclassobservers.SPDTNumericClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.classifiers.trees.HoeffdingTree;
import moa.classifiers.trees.RandomHoeffdingTree;
import moa.core.AutoExpandVector;
import moa.core.sketches.MergeableHistogram;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ThreadLocalRandom;

public class DAGLearningNode extends RandomHoeffdingTree.RandomLearningNode {

  private HoeffdingDAG tree;

  private boolean readyToSplit;

  private int tempLeft;
  private int tempRight;

  private int bestFeature;
  private double bestThreshold;

  private ClassHistogram leftHistogram;
  private ClassHistogram rightHistogram;

  private double entropy;

  public DAGLearningNode(double[] initialClassObservations, HoeffdingDAG tree) {
    super(initialClassObservations);
    leftHistogram = new ClassHistogram(initialClassObservations.length);
    rightHistogram = new ClassHistogram(initialClassObservations.length);
    this.tree = tree;
    bestFeature = 0;
    bestThreshold = 0.0;
  }


  @Override
  public void learnFromInstance(Instance inst, HoeffdingTree ht) {
    // Do the learning as usual
    super.learnFromInstance(inst, ht);
    // In addition, maintain a class histogram for the left and right children
    int childBranch = inst.valueInputAttribute(bestFeature) < bestThreshold ? 0 : 1;
    if (childBranch == 0) { // TODO: Add instance weight here instead of one
      leftHistogram.addOne(inst.classIndex());
    } else {
      rightHistogram.addOne(inst.classIndex());
    }
  }

  /**
   * Finds the best feature and threshold combination for the node, given all the other parent nodes.
   * @param parentNodes The row of parent nodes that are currently the active learning leafs.
   * @return True if the best feature/threshold combination changed, false otherwise.
   */
  public boolean findThreshold(ArrayList<DAGLearningNode> parentNodes) {
    ThresholdEntropyErrorFunction errorFunction = new ThresholdEntropyErrorFunction(
        parentNodes, this);

    errorFunction.initializeHistograms();

    double previousError = errorFunction.error();

    errorFunction.resetHistograms();

    // TODO: Reset right/eleft histograms

    boolean changed = false;

    // Iterate over all sampled features in to see if a better feature/threshold combination exists
    for (int localAttIndex = 0; localAttIndex < this.numAttributes - 1; localAttIndex++) {
      int overallAttIndex = this.listAttributes[localAttIndex];
//      assert overallAttIndex == localAttIndex : "Not sure if these should be equal actually.";
      // tvas: To get a first working version we will select a threshold randomly within the observed values
      // todo: later we'll need to optimize the threshold selection

      // To get candidate split points, we get the overall distribution of the attribute
      // disreagarding the class, and create a uniform histogram out of that.
      // The bin borders give us the candidate split points
      SPDTNumericClassObserver obs = (SPDTNumericClassObserver) attributeObservers.get(overallAttIndex);

      AutoExpandVector<MergeableHistogram> histsPerClass = obs.getAttHistPerClass();

      MergeableHistogram mergedHists = mergeHists(histsPerClass);

      double[] splitPoints = mergedHists.uniform(tree.numSplitPoints);

      // Select a single random split point
      double newSplitPoint = splitPoints[ThreadLocalRandom.current().nextInt(splitPoints.length)];

      // Now let's figure out how that would change the overall entropy of the tree.
      for (int k = 0; k < histsPerClass.size(); k++) {
        // For each class, get the new number of points at each side of the split
        MergeableHistogram classHist = histsPerClass.get(k);
        double samplesLeft = classHist.getSum(newSplitPoint);
        double samplesRight = classHist.getTotalCount() - samplesLeft;
        // TODO: Currently set will re-calculate entropies for every call (every class)
        // todo: I think it would be enough to only do that once at the end of this loop
        // Update the class histogram counts, and the entropies
        errorFunction.setClassCounts(k, (int) samplesLeft, (int) samplesRight); // TODO: Counts should be doubles
      }

      // Now let's recalculate the error/entropy
      double newError = errorFunction.error();

      // If the new splitpoint improves upon the last, we keep it as the new best suggestion
      if (newError < previousError) {
        bestFeature = overallAttIndex;
        bestThreshold = newSplitPoint;
        previousError = newError;
        changed = true;
      }
    }
    // TODO: Do we only do this if we have made a change?
    if (changed) {
      leftHistogram = errorFunction.getLeftEntropyHistogram().toClassHistogram();
      rightHistogram = errorFunction.getRightEntropyHistogram().toClassHistogram();
    }
    return changed;
  }

  private MergeableHistogram mergeHists(List<MergeableHistogram> histList) {
    MergeableHistogram mergedHist = null;
    for (MergeableHistogram hist : histList) {
      if (mergedHist == null) {
        mergedHist = hist;
        continue;
      }
      mergedHist = mergedHist.merge(hist);
    }
    return mergedHist;
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
    if (bestSplitSuggestionOptional.isPresent()) {
      readyToSplit = true;
    }

    return bestSplitSuggestionOptional;
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

  public ClassHistogram getClassHistogram() {
    AutoExpandVector<Integer> data = new AutoExpandVector<>(observedClassDistribution.numValues());
    for (int i = 0; i < data.size(); i++) {
      data.set(i, (int) observedClassDistribution.getValue(i));
    }
    return new ClassHistogram(data);
  }

}
