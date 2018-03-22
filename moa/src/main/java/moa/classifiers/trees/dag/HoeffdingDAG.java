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

import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.SPDTNumericClassObserver;
import moa.classifiers.core.conditionaltests.NumericAttributeBinaryTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.classifiers.trees.RandomHoeffdingTree;
import moa.core.AutoExpandVector;

import java.util.*;

public class HoeffdingDAG extends RandomHoeffdingTree {
  public FloatOption growthRate = new FloatOption("growthRate",
      'u', "The rate at which we allow DAG width levels to grow. 2 is equivalent to a tree.",
      1.5, 1.0 + Double.MIN_VALUE, 2.0);

  public int numSplitPoints = 10; // TODO: Add as option
  private int maxLevel = 10;
  private int maxWidth = 256;
  private int maxIterations = 1;
  private int maxBins = 64;
  private int currentLevel = 0;
  private int weightSinceLastLevel = 0;

  Set<ActiveLearningNode> readyNodes = new HashSet<>();

  SplitCriterion activeSplitCriterion;

  // tvas: This could perhaps be tied into the nodes themselves, but let's try it this way first
  private int readyToSplit; // Keeps count of how many nodes are ready to split at the current learning row
  private ArrayList<DAGLearningNode> learningRow; // All the learning nodes at the current bottom level of the DAG
  // tvas: This map is a temp solution.
  // 1: We prolly want to update the best suggestion anyway
  // 2: I don't like mapping from node to attribute suggestion
  private HashMap<ActiveLearningNode, AttributeSplitSuggestion> nodeToSplitSuggestion;


  @Override
  public void resetLearningImpl() {
    super.resetLearningImpl();
    learningRow = new ArrayList<>();
    nodeToSplitSuggestion = new HashMap<>();
    readyToSplit = 0;
    this.binarySplitsOption.setValue(true);
    activeSplitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
  }

  @Override
  public void trainOnInstanceImpl(Instance inst) {
    weightSinceLastLevel++;
    if (this.treeRoot == null) {
      this.treeRoot = newLearningNode();
      learningRow.add((DAGLearningNode) this.treeRoot);
      this.activeLeafNodeCount = 1;
    }
    FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1);
    Node leafNode = foundNode.node;

    if (leafNode instanceof LearningNode) {
      LearningNode learningNode = (LearningNode) leafNode;

      learningNode.learnFromInstance(inst, this);
      if (this.growthAllowed && (learningNode instanceof ActiveLearningNode)) {
        ActiveLearningNode activeLearningNode = (ActiveLearningNode) learningNode;
        double weightSeen = activeLearningNode.getWeightSeen();

        if (weightSeen - activeLearningNode.getWeightSeenAtLastSplitEvaluation() >= this.gracePeriodOption.getValue()) {
          Optional<AttributeSplitSuggestion> bestSplitOptional = ((DAGLearningNode) activeLearningNode).checkForSplit() ;
          if (bestSplitOptional.isPresent()) {
            readyNodes.add(activeLearningNode);
          }

          // tvas: When should we instantiate the next level of nodes?
          if (readyNodes.size() / learningRow.size() > 0.5 || weightSinceLastLevel > 25000) { // TODO: Placeholder, need to find a reasonable criterion!!
            weightSinceLastLevel = 0;
            readyNodes.clear();
            splitLearningRow();
          }
        }
      }
    }
    if (this.trainingWeightSeenByModel % this.memoryEstimatePeriodOption.getValue() == 0
        && !disableMemoryManagement.isSet()) {
      estimateModelByteSizes();
    }
  }

  private void splitLearningRow() { //TODO: Do I want the parent row and child count as arguments here?
    System.out.println("Splitting level: " + currentLevel++);
    System.out.println("Level leafs: " + learningRow.size());
    int iterations = 0;
    boolean change = false;
    int numChildren = (int) Math.max(learningRow.size() * growthRate.getValue(), 2); // At least two children for root
    boolean isTreeLevel = numChildren == learningRow.size() * 2;
    // TODO: Handle pure nodes: They don't need threshold, and should have left == right child

    // TODO: Sort the learning row decreasing by their entropy

    // Initialize new level
    // Do a base assignment of children to parent nodes
    int vChildren = 0;
    for (DAGLearningNode dagNode : learningRow) {
      // TODO: C++ does this assignment in reverse, why? --> Explained in 3.1.1: high entropy parents should not have common children
      dagNode.setTempLeft(vChildren++ % numChildren);
      dagNode.setTempRight(vChildren++ % numChildren);
    }


    do {
      // Find best feature/threshold for each node
      for (DAGLearningNode dagNode : learningRow) {
        if (dagNode.findThreshold(learningRow)) {
          change = true;
        }
      }
      // If this level is a tree level, the child assignments need no change, so we can break.
      if (isTreeLevel) {
        break;
      }
      // Find best assignment of children for each node
      for (DAGLearningNode dagNode : learningRow) {
        if (dagNode.findRightChildAssignment(learningRow, numChildren)) {
          change = true;
        }
        if (dagNode.findLeftChildAssignment(learningRow, numChildren)) {
          change = true;
        }
      }
      iterations++;
    } while (iterations < maxIterations && change);

    // TODO: Check if adding the new row improves the overall tree entropy. If not, don't create the row

    // Create the child nodes
    AutoExpandVector<DAGLearningNode> childNodes = new AutoExpandVector<>(numChildren);
    // Keep track of children without any parents assigned
    boolean[] noParentNode = new boolean[numChildren];
    Arrays.fill(noParentNode, true);

    // Assign each parent to their child nodes
    for (DAGLearningNode current : learningRow) {
      int leftNodeIndex = current.getTempLeft();
      int rightNodeIndex = current.getTempRight();

      // Create a new split node based on the current parent
      NumericAttributeBinaryTest attTest = new NumericAttributeBinaryTest(
          current.getBestFeature(), current.getBestThreshold(), true);
      SplitNode splitNode = newSplitNode(attTest, current.getObservedClassDistribution(), 2);

      // Get the (potentially) existing left and right child nodes and update their dists as necessary
      DAGLearningNode leftChild = childNodes.get(leftNodeIndex);
      leftChild = createNodeFromExisting(leftChild, current.getLeftHistogram().toArray());
      DAGLearningNode rightChild = childNodes.get(rightNodeIndex);
      rightChild = createNodeFromExisting(rightChild, current.getRightHistogram().toArray());
      // Update the collection of child nodes, this way we can retrieve and update them later
      childNodes.set(leftNodeIndex, leftChild);
      childNodes.set(rightNodeIndex, rightChild);
      // Set the created child nodes as the split node's children
      splitNode.setChild(0, leftChild);
      splitNode.setChild(1, rightChild);
      noParentNode[leftNodeIndex] = false;
      noParentNode[rightNodeIndex] = false;
      if (treeRoot == current) {
        treeRoot = splitNode;
      }
    }

    // tvas: There's a chance a threshold is selected s.t. no points reach one of the children.
    // We might want to merge the children in that case

    // Replace the current learning row with the newly created child nodes.
    activeLeafNodeCount = numChildren;
    decisionNodeCount += learningRow.size();
    learningRow.clear();
    for (int i = 0; i < numChildren; i++) {
      if (!noParentNode[i]) {
        // Children with no parents should drop off and be GCed.
        // tvas: How to verify? Any chance of lingering pointers?
        // TODO: The get here might be problematic, because the indicies for learningRow are given by the
        // current.getTempLeft/Right need to ensure those are [0, numChildren - 1] as well
        learningRow.add(childNodes.get(i));
      }
    }

  }

  private DAGLearningNode createNodeFromExisting(DAGLearningNode existingChild, double[] currentDistribution) {
    double[] combinedDistribution;
    if (existingChild == null) {
      combinedDistribution = currentDistribution;
    } else {
      double[] previousDistribution = existingChild.getObservedClassDistribution();
      assert previousDistribution.length == currentDistribution.length;
      combinedDistribution = new double[previousDistribution.length];
      for (int i = 0; i < previousDistribution.length; i++) {
        combinedDistribution[i] = previousDistribution[i] + currentDistribution[i];
      }
    }
    // TODO: Checking what happens if we don't init dists
    return new DAGLearningNode(new double[currentDistribution.length],this);
  }

  public AttributeSplitSuggestion[] getSplitSuggestions(ActiveLearningNode node) {
    SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
    AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this);
    Arrays.sort(bestSplitSuggestions);

    return bestSplitSuggestions;
  }

  @Override
  protected LearningNode newLearningNode(double[] initialClassObservations) {
    return new DAGLearningNode(initialClassObservations, this);
  }

  @Override
  protected AttributeClassObserver newNominalClassObserver() {
    throw new UnsupportedOperationException("No support for nominal features yet!");
  }

  @Override
  protected AttributeClassObserver newNumericClassObserver() {
    return new SPDTNumericClassObserver();
  }


  @Override
  public String toString() {
    return super.toString();
  }
}
