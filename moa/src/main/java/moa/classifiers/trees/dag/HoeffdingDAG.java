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
import moa.classifiers.trees.ARFHoeffdingTree;
import moa.classifiers.trees.RandomHoeffdingTree;
import moa.core.AutoExpandVector;
import org.apache.commons.math3.util.Pair;

import java.util.*;

public class HoeffdingDAG extends ARFHoeffdingTree {
  public FloatOption growthRate = new FloatOption("growthRate",
      'u', "The rate at which we allow DAG width levels to grow. 2 is equivalent to a tree.",
      1.5, 1.0 + Double.MIN_VALUE, 2.0);
  // Enum to keep track of which branch of parent a child belongs to
  public enum Branch {
    LEFT, RIGHT, NULL
  }

  // TODO: Add as options
  public int numSplitPoints = 10;
  private int maxLevel = 10;
  private int maxWidth = 256;
  private int maxIterations = 10; // TODO: GREEDY ALGO DOES NOT SEEM TO CONVERGE, REGARDLESS OF NUMBER OF ITERS!!
  private int maxBins = 64;

  private int currentLevel = 0;

  private Set<DAGLearningNode> readyNodes = new HashSet<>();

  SplitCriterion activeSplitCriterion;

  private ArrayList<DAGLearningNode> learningRow; // All the learning nodes at the current bottom level of the DAG


  @Override
  public void resetLearningImpl() {
    super.resetLearningImpl();
    learningRow = new ArrayList<>();
    this.binarySplitsOption.setValue(true);
    activeSplitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
  }

  @Override
  public void trainOnInstanceImpl(Instance inst) {
    if (this.treeRoot == null) {
      this.treeRoot = new DAGLearningNode(new double[0], this, null, Branch.NULL);
      learningRow.add((DAGLearningNode) this.treeRoot);
      this.activeLeafNodeCount = 1;
    }
    FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1);
    Node leafNode = foundNode.node;

    if (leafNode instanceof LearningNode) {
      DAGLearningNode dagNode = (DAGLearningNode) leafNode;
      dagNode.learnFromInstance(inst, this);

      if (growthAllowed) {
        double weightSeen = dagNode.getWeightSeen();
        if (weightSeen - dagNode.getWeightSeenAtLastSplitEvaluation() >= gracePeriodOption.getValue()) {

          if (!dagNode.isReadyToSplit()) {
            Optional<AttributeSplitSuggestion> bestSplitOptional = dagNode.checkForSplit();
            dagNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
            if (bestSplitOptional.isPresent()) {
              readyNodes.add(dagNode);
            }
          }
          // tvas: When should we instantiate the next level of nodes?
          if (readyNodes.size() / learningRow.size() > 0.5) { // TODO: Placeholder, need to find a reasonable criterion!!
            splitLearningRow();
            readyNodes.clear();
          }
        }
      }
    }
    if (trainingWeightSeenByModel % memoryEstimatePeriodOption.getValue() == 0
        && !disableMemoryManagement.isSet()) {
      estimateModelByteSizes();
    }
  }

//  private void optimizeLevel(ArrayList<DAGLearningNode> row, int maxIterations) {
//    // TODO: A function that takes an existing row of nodes, and optimizes their splits/assignments based
      // on their children's actual stats.
//  }

  private void splitLearningRow() { //TODO: Do I want the parent row and child count as arguments here?
    System.out.println("Splitting level: " + currentLevel++);
    System.out.println("Level leafs: " + learningRow.size());
    System.out.println("Nodes ready to split: " + readyNodes.size());

    learningRow.removeIf(curNode -> !readyNodes.contains(curNode));
    // TODO: Turn non-ready nodes to inactive, or keep them in the learning row?
    assert learningRow.size() > 0 : "There should always be at least one learning node!";

    // Ensure we get two children for the root
    int numChildren = (int) Math.ceil(Math.max(learningRow.size() * growthRate.getValue(), 2));
    boolean isTreeLevel = numChildren == learningRow.size() * 2;
    // TODO: Sort the learning row decreasing by their entropy, iterate in that order

    // Initialize new level
    // Do an initial assignment of children to parent nodes
    int vChildren = 0;
    for (DAGLearningNode dagNode : learningRow) {
      if (dagNode.getBestFeature() == -1) {
        if (dagNode.observedClassDistributionIsPure()) {
          dagNode.setBestFeature(0);
        } else {
          dagNode.findThreshold(learningRow);
        }
        assert dagNode.getBestFeature() != -1;
      }
      dagNode.updateLeftRightHistogram(); // TODO: Necessary?
      // TODO: C++ does this assignment in reverse, why? --> Explained in 3.1.1: high entropy parents should not have common children
      if (dagNode.observedClassDistributionIsPure()) {
        // We assign both child pointers to one node
        dagNode.setTempLeft(vChildren % numChildren);
        dagNode.setTempRight(vChildren++ % numChildren);
      } else {
        dagNode.setTempLeft(vChildren++ % numChildren);
        dagNode.setTempRight(vChildren++ % numChildren);
      }

    }

    int iterations = 0;
    boolean change;
    Map<Integer, Integer> thresholdChanges = new HashMap<>();
    Map<Integer, Integer> assignmentChanges = new HashMap<>();

    do {
      change = false;
      // Find best feature/threshold for each node
      for (DAGLearningNode dagNode : learningRow) {
        // The thresholds are optimized for the first time during initialization
        if (iterations == 0) {
          break; // Break out the inner for loop
        }
        if (dagNode.observedClassDistributionIsPure() || dagNode.getTempLeft() == dagNode.getTempRight()) {
          // Pure nodes don't need a threshold
          continue;
        }
        if (dagNode.findThreshold(learningRow)) {
          int count = thresholdChanges.getOrDefault(dagNode.getLeafID(), 0);
          thresholdChanges.put(dagNode.getLeafID(), count + 1);
          change = true;
        }
      }
      // If this level is a tree level, the child assignments need no change, so we can break.
      if (isTreeLevel) {
        break;
      }
      // Find best assignment of children for each node
      for (DAGLearningNode dagNode : learningRow) {
        if (dagNode.observedClassDistributionIsPure()) {
          // Find optimal child assignment for both pointers
          if (dagNode.findCoherentChildNodeAssignment(learningRow, numChildren)) {
            int count = assignmentChanges.getOrDefault(dagNode.getLeafID(), 0);
            assignmentChanges.put(dagNode.getLeafID(), count + 1);
            change = true;
          }
        } else { // If the node is not pure, we separately optimize its left and right child assignment
          if (dagNode.findLeftChildAssignment(learningRow, numChildren)) {
            int count = assignmentChanges.getOrDefault(dagNode.getLeafID(), 0);
            assignmentChanges.put(dagNode.getLeafID(), count + 1);
            change = true;
          }
          if (dagNode.findRightChildAssignment(learningRow, numChildren)) {
            int count = assignmentChanges.getOrDefault(dagNode.getLeafID(), 0);
            assignmentChanges.put(dagNode.getLeafID(), count + 1);
            change = true;
          }
        }
      }
      iterations++;
    } while (iterations < maxIterations && change);
    System.out.println("Iterations until stability: " + iterations);
    System.out.println("Changes made:");
    System.out.println("Threshold changes: " + thresholdChanges);
    System.out.println("Assignment changes:" + assignmentChanges);

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
      // After each creation, update the collection of child nodes, this way we can retrieve and update them later
      DAGLearningNode leftChild = childNodes.get(leftNodeIndex);
      leftChild = createNodeFromExisting(leftChild, current.getLeftHistogram().toArray(), splitNode, Branch.LEFT);
      childNodes.set(leftNodeIndex, leftChild);

      DAGLearningNode rightChild = childNodes.get(rightNodeIndex);
      rightChild = createNodeFromExisting(rightChild, current.getRightHistogram().toArray(), splitNode, Branch.RIGHT);
      childNodes.set(rightNodeIndex, rightChild);

      // Set the created child nodes as the split node's children
      splitNode.setChild(0, leftChild);
      splitNode.setChild(1, rightChild);

      noParentNode[leftNodeIndex] = false;
      noParentNode[rightNodeIndex] = false;
      if (treeRoot == current) {
        treeRoot = splitNode;
      } else {
        // Change the pointers of all the parents for the current node to point to the newly created split node
        for (Pair<SplitNode, Branch> parentBranch : current.getParents()) {
          SplitNode parent = parentBranch.getFirst();
          Branch branchDecision = parentBranch.getSecond();
          if (branchDecision == Branch.LEFT) {
            parent.setChild(0, splitNode);
          } else {
            parent.setChild(1, splitNode);
          }
        }
      }
    }
    assert childNodes.size() == numChildren;
    // tvas: There's a chance a threshold is selected s.t. no points reach one of the children.
    // We might want to merge the children in that case

    // Replace the current learning row with the newly created child nodes.
    activeLeafNodeCount = numChildren;
    decisionNodeCount += learningRow.size();
    learningRow.clear();
    for (int i = 0; i < numChildren; i++) {
      if (!noParentNode[i]) {
        // TODO: The get here might be problematic, because the indicies for learningRow are given by the
        // current.getTempLeft/Right need to ensure those are [0, numChildren - 1] as well
        learningRow.add(childNodes.get(i));
      }
    }
  }

  private void describeTree(StringBuilder out, int indent) {
    Stack<Node> nodeStack = new Stack<>();

    nodeStack.push(treeRoot);

    while (!nodeStack.isEmpty()) {
      Node curNode = nodeStack.pop();
      if (curNode instanceof SplitNode) {
        SplitNode splitNode = (SplitNode) curNode;
        nodeStack.push(splitNode.getChild(0));
        nodeStack.push(splitNode.getChild(1));
        splitNode.describeSubtree(this, out, 2);
      } else {
        DAGLearningNode learningNode = (DAGLearningNode) curNode;
        learningNode.describeSubtree(this, out, 2);
      }
    }
  }

  private DAGLearningNode createNodeFromExisting(DAGLearningNode existingChild, double[] currentDistribution , SplitNode parent, Branch branch) {
    // TODO: Ignoring the histograms for now to check what happens if we don't init dists. FIX
    // If the node did not exist create a new one, otherwise just add to its list of parents
    if (existingChild == null) {
      existingChild = new DAGLearningNode(new double[currentDistribution.length], this, parent, branch);
    } else {
      existingChild.addParent(parent, branch);
    }
    return existingChild;

//    double[] combinedDistribution;
//    if (existingChild == null) {
//      combinedDistribution = currentDistribution;
//    } else {
//      double[] previousDistribution = existingChild.getObservedClassDistribution();
//      assert previousDistribution.length == currentDistribution.length;
//      combinedDistribution = new double[previousDistribution.length];
//      for (int i = 0; i < previousDistribution.length; i++) {
//        combinedDistribution[i] = previousDistribution[i] + currentDistribution[i];
//      }
//    }
//    return new DAGLearningNode(new double[currentDistribution.length],this);
  }

  public AttributeSplitSuggestion[] getSplitSuggestions(ActiveLearningNode node) {
    SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
    AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this);
    Arrays.sort(bestSplitSuggestions);

    return bestSplitSuggestions;
  }

  @Override
  protected LearningNode newLearningNode(double[] initialClassObservations) {
    throw new UnsupportedOperationException("We have custom new node creation for HoeffdingDAG!");
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
