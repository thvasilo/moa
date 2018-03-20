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

import java.util.ArrayList;


public class ThresholdEntropyErrorFunction {
  // The row of learning nodes (parents)
  private ArrayList<DAGLearningNode> row;

  // The specific parent we are optimizing currently
  private DAGLearningNode currentParent;

  public ThresholdEntropyErrorFunction(ArrayList<DAGLearningNode> row, DAGLearningNode currentParent) {
    this.row = row;
    this.currentParent = currentParent;
    initializeHistograms();
  }

  ClassHistogram leftHistogram;
  ClassHistogram rightHistogram;
  EntropyHistogram leftEntropyHistogram;
  EntropyHistogram rightEntropyHistogram;

  void initializeHistograms() {

    // TODO: Not sure if this will be correct, as some classes may have not been observed.
    // It might be safer to get the number of classes some other way
    int classCount = row.get(0).getObservedClassDistribution().length;

    leftHistogram.resize(classCount);
    rightHistogram.resize(classCount);
    leftEntropyHistogram.resize(classCount);
    rightEntropyHistogram.resize(classCount);

    // Compute the base histograms for all child nodes
    for (int i = 0; i < row.size(); i++) {
      DAGLearningNode otherParent = row.get(i);

      // Skip the current parent
      if (otherParent.equals(currentParent)) {
        continue;
      }

      int leftNode = otherParent.getTempLeft();
      int rightNode = otherParent.getTempRight();
      ClassHistogram otherLeftHistogram = otherParent.getLeftHistogram();
      ClassHistogram otherRightHistogram = otherParent.getRightHistogram();

      // We iterate though all the child nodes, if one of the childs of the other parents has the current parent
      // as a parent as well, we add the hisstograms of those parents to the total.
      if (leftNode == otherParent.getTempLeft()) {
        // Current's left child is other's left child as well
        leftHistogram.merge(otherLeftHistogram);
      }
      if (rightNode == otherParent.getTempRight()) {
        // Current's right child is other's right child as well
        rightHistogram.merge(otherRightHistogram);
      }
      if (leftNode == otherParent.getTempRight()) {
        // Current's left child is other's right child as well
        leftHistogram.merge(otherRightHistogram);
      }
      if (rightNode == otherParent.getTempLeft()) {
        // Current's right child is other's left child as well
        rightHistogram.merge(otherLeftHistogram);
      }
    }

    // Compute the current histograms for nodes that are children of the current parent (?)
    leftEntropyHistogram.merge(leftHistogram); // TODO: Do I want to be able to do this in one step, to avoid one merge?
    leftEntropyHistogram.merge(currentParent.getLeftHistogram());
    leftEntropyHistogram.initializeEntropies();
    rightEntropyHistogram.merge(rightHistogram);
    rightEntropyHistogram.merge(currentParent.getRightHistogram());
    rightEntropyHistogram.initializeEntropies();

  }

  /**
   * Calculates the error if we split. This function expects the local histograms to be already computed.
   */
  double error()
  {
    return  1/(leftEntropyHistogram.getMass() + rightEntropyHistogram.getMass()) *
        (leftEntropyHistogram.entropy() + rightEntropyHistogram.entropy());
  }
}
