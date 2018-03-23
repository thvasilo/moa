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
import org.apache.commons.math3.util.Pair;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class DAGLearningNode extends RandomHoeffdingTree.RandomLearningNode {

  // Each entry in this list holds one parent pointer, and the branch
  private ArrayList<Pair<HoeffdingTree.SplitNode, HoeffdingDAG.Branch>> parents;

  private HoeffdingDAG tree;

  private boolean readyToSplit;

  private int tempLeft;
  private int tempRight;

  private int bestFeature;
  private double bestThreshold;

  private ClassHistogram leftHistogram;
  private ClassHistogram rightHistogram;

  private double entropy;

  public DAGLearningNode(double[] initialClassObservations, HoeffdingDAG tree, HoeffdingTree.SplitNode parent, HoeffdingDAG.Branch parentBranch) {
    super(initialClassObservations);
    leftHistogram = new ClassHistogram(initialClassObservations.length);
    rightHistogram = new ClassHistogram(initialClassObservations.length);
    this.tree = tree;
    parents = new ArrayList<>();
    parents.add(new Pair<>(parent, parentBranch));
    bestFeature = -1;
    bestThreshold = 0.0;
  }

  public void addParent(HoeffdingTree.SplitNode parent, HoeffdingDAG.Branch parentBranch) {
    if (parents.isEmpty()) {
      parents = new ArrayList<>();
    }
    parents.add(new Pair<>(parent, parentBranch));
  }



  @Override

  public void learnFromInstance(Instance inst, HoeffdingTree ht) {
    // Do the learning as usual
    super.learnFromInstance(inst, ht);

    // In addition, maintain a class histogram for the left and right children
    if (bestFeature != -1) { // Only makes sense if we have a feature to split on TODO: What about pure nodes then?
      int childBranch = inst.valueInputAttribute(bestFeature) <= bestThreshold ? 0 : 1;
      if (childBranch == 0) { // TODO: Add instance weight here instead of one
        leftHistogram.addOne((int) inst.classValue());
      } else {
        rightHistogram.addOne((int) inst.classValue());
      }
    }

  }

  /**
   * Finds the best feature and threshold combination for the node, given all the parent nodes.
   *
   * @param parentNodes The row of parent nodes that are currently the active learning leafs.
   * @return True if the best feature/threshold combination changed, false otherwise.
   */
  public boolean findThreshold(ArrayList<DAGLearningNode> parentNodes) {
    // The error function object keeps track of the entropy statistics of the, including how the other parents affect
    // the current node's children.
    ThresholdEntropyErrorFunction errorFunction = new ThresholdEntropyErrorFunction(
        parentNodes, this);

    // This will initialize the histograms with the relevant data from all parents
    errorFunction.initializeHistograms();

    // Calculate the initial entropy/error
    double previousError = bestFeature == -1 ? Double.MAX_VALUE : errorFunction.error();
    ClassHistogram bestLeftHistogram = errorFunction.getLeftEntropyHistogram().toClassHistogram();
    ClassHistogram bestRightHistogram = errorFunction.getRightEntropyHistogram().toClassHistogram();

    // We then reset the current node's histograms. These are the ones that change as we move the node's split point.
    errorFunction.resetHistograms();

    boolean changed = false;

    // Clear the left histogram, and the right one becomes the node histogram.
    this.resetLeftRightHistogram();

    // Iterate over all sampled features to see if a better feature/threshold combination exists
    for (int localAttIndex = 0; localAttIndex < this.numAttributes - 1; localAttIndex++) {
      int overallAttIndex = this.listAttributes[localAttIndex];
      // To get candidate split points, we get the overall distribution of the attribute
      // disreagarding the class, and create a uniform histogram out of that.
      // The bin borders give us the candidate split points. See SPDT Sec. 2.2.
      SPDTNumericClassObserver obs = (SPDTNumericClassObserver) attributeObservers.get(overallAttIndex);
      AutoExpandVector<MergeableHistogram> histsPerClass = obs.getAttHistPerClass();
      MergeableHistogram mergedHists = mergeHists(histsPerClass);
      double[] splitPoints = mergedHists.uniform(tree.numSplitPoints);

      // Iterate through the possible split points
      for (double newSplitPoint : splitPoints) { // tvas: Pretty sure this could be done in parallel if careful with histograms
        // Now let's figure out how that would change the overall entropy of the tree.
        for (int k = 0; k < histsPerClass.size(); k++) {
          // For each class, get the new number of points at each side of the split
          MergeableHistogram classHist = histsPerClass.get(k);
          double samplesLeft = classHist == null ? 0 : classHist.getSum(newSplitPoint);
          double samplesRight = classHist == null ? 0 : classHist.getTotalCount() - samplesLeft;
          // Update the class histogram counts. NB: entropies are not re-calculated until the call to error(true) below
          errorFunction.setClassCounts(k, (int) samplesLeft, (int) samplesRight); // TODO: Counts should be doubles
        }

        // Now let's recalculate the error/entropy
        double newError = errorFunction.error(true);

        // If the new splitpoint is not almost equal and improves upon the last error, make it the new best suggestion
        boolean changeSignificant = true;
        if (overallAttIndex == bestFeature && Math.abs(newSplitPoint - bestThreshold) <= 1e-6) {
          changeSignificant = false;
        }
        if ((newError < previousError && changeSignificant) || bestFeature == -1) {
          bestFeature = overallAttIndex;
          bestThreshold = newSplitPoint;
          bestLeftHistogram = errorFunction.getLeftEntropyHistogram().toClassHistogram();
          bestRightHistogram = errorFunction.getRightEntropyHistogram().toClassHistogram();
          previousError = newError;
          changed = true;
        }
      }
    }

//    // TODO: Is this OK or do we need to recalculate these?
//    leftHistogram = bestLeftHistogram;
//    rightHistogram = bestRightHistogram;
    updateLeftRightHistogram();

    if (bestFeature == -1) {
      System.out.println("Node " + this.toString() + " failed to get a new split feature.");
      System.out.println("It has seen " + getWeightSeen() + " learning weight.");
    }

    return changed;
  }

  public boolean findCoherentChildNodeAssignment(ArrayList<DAGLearningNode> parentNodes, int childNodeCount) {
    AssignmentEntropyErrorFunction errorFunction = new AssignmentEntropyErrorFunction(
        parentNodes, this, childNodeCount);

    errorFunction.initializeHistograms();

    // Save the current selection
    int selectedRight = getTempRight();
    int selectedLeft = getTempLeft();

    double oldEntropy = errorFunction.error();
    double currentEntropy = 0;
    boolean changed = false;

    // Test all possible assignments for the common child pointers
    for (int current = 0; current < childNodeCount; current++) {
      // Make a temp assignment
      setTempRight(current);
      setTempLeft(current);

      // Get the error for this assignment
      currentEntropy = errorFunction.error();

      // If it's better than the current best, change the right temp node selection
      if (currentEntropy < oldEntropy) {
        selectedRight = current;
        selectedLeft = current;
        oldEntropy = currentEntropy;
        changed = true;
      }
    }

    setTempRight(selectedRight);
    setTempLeft(selectedLeft);

    return changed;
  }

  // TODO: Assignment functions duplicate code, could probably be done by passing setTemp/getTemp<Right/Left>
  // todo: as lambdas, but this will do for now

  public boolean findRightChildAssignment(ArrayList<DAGLearningNode> parentNodes, int childNodeCount) {
    AssignmentEntropyErrorFunction errorFunction = new AssignmentEntropyErrorFunction(
        parentNodes, this, childNodeCount);

    errorFunction.initializeHistograms();

    // Save the current selection
    int selectedRight = getTempRight();

    double oldEntropy = errorFunction.error();
    double currentEntropy = 0;
    boolean changed = false;

    // Test all possible assignments for the right child
    for (int curRight = 0; curRight < childNodeCount; curRight++) {
      // Make a temp assignment of the right node
      setTempRight(curRight);

      // Get the error for this assignment
      currentEntropy = errorFunction.error();

      // If it's better than the current best, change the right temp node selection
      if (currentEntropy < oldEntropy) {
        selectedRight = curRight;
        oldEntropy = currentEntropy;
        changed = true;
      }
    }

    setTempRight(selectedRight);

    return changed;
  }

  public boolean findLeftChildAssignment(ArrayList<DAGLearningNode> parentNodes, int childNodeCount) {
    AssignmentEntropyErrorFunction errorFunction = new AssignmentEntropyErrorFunction(
        parentNodes, this, childNodeCount);

    errorFunction.initializeHistograms();

    // Save the current selection
    int selectedLeft = getTempLeft();

    double oldEntropy = errorFunction.error();
    double currentEntropy = 0;
    boolean changed = false;

    // Test all possible assignments for the right child
    for (int curleft = 0; curleft < childNodeCount; curleft++) {
      // Make a temp assignment of the right node
      setTempLeft(curleft);

      // Get the error for this assignment
      currentEntropy = errorFunction.error();

      // If it's better than the current best, change the right temp node selection
      if (currentEntropy < oldEntropy) {
        selectedLeft = curleft;
        oldEntropy = currentEntropy;
        changed = true;
      }
    }

    setTempLeft(selectedLeft);

    return changed;
  }


  /**
   * Reset's the node's histograms by clearing the left histogram, and the right histogram becoming the node's histogram.
   */
  private void resetLeftRightHistogram() {
    // TODO: Ensure this works as expected
    ClassHistogram nodeHist = getClassHistogram();

    leftHistogram.reset();
    rightHistogram.reset();
    rightHistogram = rightHistogram.merge(nodeHist);
  }

  public void updateLeftRightHistogram() {

    leftHistogram.reset();
    rightHistogram.reset();

    if (this.observedClassDistributionIsPure()) {
      return;
    }

    SPDTNumericClassObserver obs = (SPDTNumericClassObserver) attributeObservers.get(bestFeature);
    AutoExpandVector<MergeableHistogram> histsPerClass = obs.getAttHistPerClass();

    for (int k = 0; k < histsPerClass.size(); k++) {
      // For each class, get the new number of points at each side of the split
      MergeableHistogram classHist = histsPerClass.get(k);
      double samplesLeft = classHist == null ? 0 : classHist.getSum(bestThreshold);
      double samplesRight = classHist == null ? 0 : classHist.getTotalCount() - samplesLeft;
      // Update the class histogram counts.
      leftHistogram.set(k, samplesLeft);
      rightHistogram.set(k, samplesRight);
    }
  }

  private static MergeableHistogram mergeHists(List<MergeableHistogram> histList) {
    // TODO: Ensure this works as expected. Also I don' think this belongs here, move to MergableHistogram or something
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
   *
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
  // TODO: This process has the issue: The decision to split a node
  // ignores the distribution of the potential child's other parents.
  public AttributeSplitSuggestion[] getBestSplitSuggestions(
      SplitCriterion criterion, HoeffdingTree ht) {
    List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<>();
    double[] preSplitDist = this.observedClassDistribution.getArrayCopy();

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

  public void setTempLeft(int tempLeft) {
    this.tempLeft = tempLeft;
  }

  public void setTempRight(int tempRight) {
    this.tempRight = tempRight;
  }

  public ClassHistogram getLeftHistogram() {
    return leftHistogram;
  }

  public ClassHistogram getRightHistogram() {
    return rightHistogram;
  }

  public ClassHistogram getClassHistogram() {
    DefaultVector data = new DefaultVector(observedClassDistribution.numValues());
    for (int i = 0; i < observedClassDistribution.numValues(); i++) {
      data.set(i, observedClassDistribution.getValue(i));
    }
    return new ClassHistogram(data);
  }

  public int getBestFeature() {
    return bestFeature;
  }

  public void setBestFeature(int bestFeature) {
    this.bestFeature = bestFeature;
  }

  public double getBestThreshold() {
    return bestThreshold;
  }

  public ArrayList<Pair<HoeffdingTree.SplitNode, HoeffdingDAG.Branch>> getParents() {
    return parents;
  }
}
