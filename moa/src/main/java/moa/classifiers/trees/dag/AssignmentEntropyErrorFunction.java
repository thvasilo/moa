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

import moa.core.AutoExpandVector;

import java.util.ArrayList;

public class AssignmentEntropyErrorFunction {

  private ArrayList<DAGLearningNode> parentRow;

  private DAGLearningNode currentParent;

  // Used to pre-compute class distributions for each child node, coming from all parents except current
  private ClassHistogram[] histograms;
  private double[] entropies; // The entropy of each child node's class distribution, excluding the data of the current parent
  private long dataCount; // The total mass i.e. sum of data point weights, excluding those present at the current parent

  private int childNodeCount;

  public AssignmentEntropyErrorFunction(ArrayList<DAGLearningNode> parentRow, DAGLearningNode parent, int childNodeCount) {
    this.parentRow = parentRow;
    this.currentParent= parent;
    this.childNodeCount = childNodeCount;
    initializeHistograms();
  }

  public void initializeHistograms() {
    int numClasses = currentParent.getObservedClassDistribution().length;

    dataCount = 0;
    // We build up a histogram for every child node
    histograms = new ClassHistogram[childNodeCount];
    for (int i = 0; i < histograms.length; i++) {
      histograms[i] = new ClassHistogram();
      histograms[i].resize(numClasses);
    }

    // Compute the histograms for all child nodes
    for (DAGLearningNode dagNode : parentRow) {
      int leftNode = dagNode.getTempLeft();
      int rightNode = dagNode.getTempRight();
      ClassHistogram leftHist = dagNode.getLeftHistogram();
      ClassHistogram rightHist = dagNode.getRightHistogram();

      dataCount += leftHist.getMass() + rightHist.getMass();

      // Skip if the current dagNode is the current parent
      if (currentParent == dagNode) {
        continue;
      }

      histograms[leftNode] = histograms[leftNode].merge(leftHist);
      histograms[rightNode] = histograms[rightNode].merge(rightHist);
    }

    // Calculate the entropies based on the histograms we just built

    entropies = new double[childNodeCount];

    for (int i = 0; i < childNodeCount; i++) {
      entropies[i] = histograms[i].calculateEntropy();
    }
  }

  /**
   * Calculate the total entropy according to the current assignment of children for the current parent
   * @return The total entropy of the tree, if we realized the current parent's temp children.
   */
  public double error() {
    double errorSum = 0;

    for (int i = 0; i < childNodeCount; i++) {
      // If the child is the left child and not the right child of the current parent
      if (i == currentParent.getTempLeft() && i != currentParent.getTempRight())
      {
        ClassHistogram leftHistogram = currentParent.getLeftHistogram();
        // The the current parent contributes entropy according to its left histogram
        errorSum += (leftHistogram.getMass() + histograms[i].getMass())/dataCount *
            ClassHistogram.calculateCombinedEntropy(leftHistogram, histograms[i]);
      }
      // If the child is the right child and not the left child of the current parent
      else if (i == currentParent.getTempRight() && i != currentParent.getTempLeft())
      {
        ClassHistogram rightHistogram = currentParent.getRightHistogram();

        errorSum += (rightHistogram.getMass() + histograms[i].getMass())/dataCount *
            ClassHistogram.calculateCombinedEntropy(rightHistogram, histograms[i]);
      }
      // If the child is the left child and the right child of the current parent. Happens when we have pure nodes
      else if (i == currentParent.getTempRight() && i == currentParent.getTempLeft())
      {
        ClassHistogram leftHistogram = currentParent.getLeftHistogram();
        ClassHistogram rightHistogram = currentParent.getRightHistogram();

        double totalMass = rightHistogram.getMass() + histograms[i].getMass() + leftHistogram.getMass();
        errorSum += totalMass/dataCount * ClassHistogram.calculateCombinedEntropy(
            rightHistogram, leftHistogram, histograms[i]);
      }
      else
      {
        errorSum += histograms[i].getMass()/dataCount * entropies[i];
      }
    }

    return errorSum;
  }
}
