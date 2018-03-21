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

  private int numClasses;
  private ClassHistogram leftHistogram;
  private ClassHistogram rightHistogram ;
  private EntropyHistogram leftEntropyHistogram;
  private EntropyHistogram rightEntropyHistogram;


  public ThresholdEntropyErrorFunction(ArrayList<DAGLearningNode> row, DAGLearningNode currentParent) {
    this.row = row;
    this.currentParent = currentParent;
    numClasses = currentParent.getObservedClassDistribution().length;
    leftHistogram = new ClassHistogram(numClasses);
    rightHistogram = new ClassHistogram(numClasses);
    leftEntropyHistogram = new EntropyHistogram(numClasses);
    rightEntropyHistogram = new EntropyHistogram(numClasses);
    initializeHistograms();
  }

  public void initializeHistograms() {


    leftHistogram.resize(numClasses);
    rightHistogram.resize(numClasses);
    leftEntropyHistogram.resize(numClasses);
    rightEntropyHistogram.resize(numClasses);

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
        leftHistogram = leftHistogram.merge(otherLeftHistogram);
      }
      if (rightNode == otherParent.getTempRight()) {
        // Current's right child is other's right child as well
        rightHistogram = rightHistogram.merge(otherRightHistogram);
      }
      if (leftNode == otherParent.getTempRight()) {
        // Current's left child is other's right child as well
        leftHistogram = leftHistogram.merge(otherRightHistogram);
      }
      if (rightNode == otherParent.getTempLeft()) {
        // Current's right child is other's left child as well
        rightHistogram = rightHistogram.merge(otherLeftHistogram);
      }
    }

    // Compute the current histograms for nodes that are children of the current parent (?)
    leftEntropyHistogram = new EntropyHistogram(leftHistogram.merge(currentParent.getLeftHistogram()));
    rightEntropyHistogram = new EntropyHistogram(rightHistogram.merge(currentParent.getRightHistogram()));
  }

  void resetHistograms() {
    rightEntropyHistogram.reset();

    leftEntropyHistogram = new EntropyHistogram(leftHistogram);
    rightEntropyHistogram = new EntropyHistogram(rightHistogram.merge(currentParent.getClassHistogram()));
  }

  /**
   * Calculates the error if we split. This function expects the local histograms to be already computed.
   */
  public double error()
  {
    return  1/(leftEntropyHistogram.getMass() + rightEntropyHistogram.getMass()) *
        (leftEntropyHistogram.entropy() + rightEntropyHistogram.entropy());
  }

  /**
   * Moves one training example from the right to the left histogram
   */
  public void move(int classLabel)
  {
    leftEntropyHistogram.addOne(classLabel);
    rightEntropyHistogram.subOne(classLabel);
  }

  public void setClassCounts(int classLabel, int leftCount, int rightCount) {
    leftEntropyHistogram.set(classLabel, leftCount);
    rightEntropyHistogram.set(classLabel, rightCount);
  }

  public EntropyHistogram getLeftEntropyHistogram() {
    return leftEntropyHistogram;
  }

  public EntropyHistogram getRightEntropyHistogram() {
    return rightEntropyHistogram;
  }
}
