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
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.classifiers.trees.HoeffdingTree;
import moa.classifiers.trees.RandomHoeffdingTree;

import java.util.*;

public class HoeffdingDAG extends RandomHoeffdingTree {
  public FloatOption growthRate = new FloatOption("growthRate",
      'u', "The rate at which we allow DAG width levels to grow. 2 is equivalent to a tree.",
      1.5, 1.0 + Double.MIN_VALUE, 2.0);

  public int numSplitPoints = 10; // TODO: Add as option
  private int maxLevel = 10;
  private int maxWidth = 256;
  private int maxIterations = 10;
  private int maxBins = 50; // From SPDT paper
  private boolean useOldBestSplit = true;
  SplitCriterion activeSplitCriterion;

  // tvas: This could perhaps be tied into the nodes themselves, but let's try it this way first
  int readyToSplit; // Keeps count of how many nodes are ready to split at the current learning row
  ArrayList<DAGLearningNode> learningRow; // All the learning nodes at the current bottom level of the DAG
  // tvas: This map is a temp solution.
  // 1: We prolly want to update the best suggestion anyway
  // 2: I don't like mapping from node to attribute suggestion
  HashMap<ActiveLearningNode, AttributeSplitSuggestion> nodeToSplitSuggestion;


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
    if (this.treeRoot == null) {
      this.treeRoot = newLearningNode();
      learningRow.add((DAGLearningNode) this.treeRoot);
      this.activeLeafNodeCount = 1;
    }
    FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1);
    Node leafNode = foundNode.node;
    if (leafNode == null) {
      leafNode = newLearningNode();
      foundNode.parent.setChild(foundNode.parentBranch, leafNode);
      this.activeLeafNodeCount++;
    }
    if (leafNode instanceof LearningNode) {
      LearningNode learningNode = (LearningNode) leafNode;

      learningNode.learnFromInstance(inst, this);
      if (this.growthAllowed && (learningNode instanceof ActiveLearningNode)) {
        ActiveLearningNode activeLearningNode = (ActiveLearningNode) learningNode;
        double weightSeen = activeLearningNode.getWeightSeen();

        if (weightSeen - activeLearningNode.getWeightSeenAtLastSplitEvaluation() >= this.gracePeriodOption.getValue()) {
          Optional<AttributeSplitSuggestion> bestSplitOptional = ((DAGLearningNode) activeLearningNode).checkForSplit() ;
          if (bestSplitOptional.isPresent()) {
            readyToSplit += 1;
            nodeToSplitSuggestion.put(activeLearningNode, bestSplitOptional.get());
          }
//          attemptToSplit(activeLearningNode, foundNode.parent, foundNode.parentBranch);
          // tvas: When should we instantiate the next level of nodes?
          if (readyToSplit / learningRow.size() > 0.5) { // TODO: Placeholder, need to find a reasonable criterion!!
            splitLearningRow();
            readyToSplit = 0;
          }
          activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
        }
      }
    }
    if (this.trainingWeightSeenByModel % this.memoryEstimatePeriodOption.getValue() == 0
        && !disableMemoryManagement.isSet()) {
      estimateModelByteSizes();
    }
  }

  private void splitLearningRow() { //TODO: Do I want the parent row and child count as arguments here?

    int iterations = 0;
    boolean change = false;
    do {
      // Find threshold
      for (Node node : learningRow) {
        DAGLearningNode dagNode = (DAGLearningNode) node;
        change = dagNode.findThreshold(learningRow);
      }
      // Find best assignment
      // TODO
    } while (iterations < maxIterations && change);

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
