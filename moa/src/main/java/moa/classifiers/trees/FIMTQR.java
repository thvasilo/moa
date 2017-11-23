package moa.classifiers.trees;
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

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.bigml.histogram.*;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;

public class FIMTQR extends FIMTDD {

  public IntOption numBins = new IntOption(
      "numBins", 'b', "Number of bins to use at leaf histograms",
      100, 1, Integer.MAX_VALUE);

  public FIMTQR(int numBins) {
    this.numBins.setValue(numBins);
  }

  // Thin interface to define common function for leaves
  // Idea stolen from Scala traits and here: https://stackoverflow.com/a/21824485/209882
  // tvas: We could generalize this and allow any tree regressor do this type of learning, I think.
  public interface withHistogram {
    // Should we define the Histogram variable here? Is it used by SplitNodes?
    Histogram getPredictionHistogram(Instance instance);
  }

  public static class QRLeafNode extends LeafNode implements withHistogram{

    private Histogram labelHistogram;

    /**
     * Create a new LeafNode
     *
     * @param tree
     */
    public QRLeafNode(FIMTQR tree) {
      super(tree);
      labelHistogram = new Histogram(tree.numBins.getValue());
    }

    @Override
    public void learnFromInstance(Instance inst, boolean growthAllowed) {
      super.learnFromInstance(inst, growthAllowed);
      try {
        labelHistogram.insert(inst.classValue());
      } catch (MixedInsertException e) {
        e.printStackTrace();
      }
    }

    @Override
    public Histogram getPredictionHistogram(Instance instance) {
      return labelHistogram;
    }
  }

  public static class QRSPlitNode extends SplitNode implements withHistogram{

    /**
     * Create a new SplitNode
     *
     * @param splitTest
     * @param tree
     */
    public QRSPlitNode(InstanceConditionalTest splitTest, FIMTDD tree) {
      super(splitTest, tree);
    }

    public Histogram getPredictionHistogram(Instance instance) {
      Node curNode = children.get(splitTest.branchForInstance(instance));
      assert curNode instanceof withHistogram;
      return ((withHistogram) curNode).getPredictionHistogram(instance);
    }
  }


  @Override
  protected LeafNode newLeafNode() {
    maxID++;
    return new QRLeafNode(this);
  }

  @Override
  protected SplitNode newSplitNode(InstanceConditionalTest splitTest) {
    maxID++;
    return new QRSPlitNode(splitTest, this);
  }

  public Histogram getPredictionHistogram(Instance instance) {
    return ((withHistogram)treeRoot).getPredictionHistogram(instance);
  }

}
