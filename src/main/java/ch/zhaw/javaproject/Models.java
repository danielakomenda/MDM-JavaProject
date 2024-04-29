/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ch.zhaw.javaproject;

import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;

import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import ai.djl.nn.*;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.*;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;


/** A helper class loads and saves model. */
public final class Models {

    // Number of Classification Labels
    public static final int NUM_OF_OUTPUT = 8;

    // Height and Width of Images
    public static final int IMAGE_HEIGHT = 128;
    public static final int IMAGE_WIDTH = 128;

    // Model-Name
    public static final String MODEL_NAME = "fruitclassifier";

    private Models() {}

    public static Model getModel() {
        // Create new instance of an empty model
        Model model = Model.newInstance(MODEL_NAME);
        // Create Model Block
        Block modelBlock = getCustomModel();
        // Set the model block
        model.setBlock(modelBlock);
        return model;
    }

    // Save model synset
    public static void saveSynset(Path modelDir, List<String> synset) throws IOException {
        Path synsetFile = modelDir.resolve("synset.txt");
        try (Writer writer = Files.newBufferedWriter(synsetFile)) {
            writer.write(String.join("\n", synset));
        }
    }

    // Create the Neural Network
    public static Block getCustomModel() {
        SequentialBlock block = new SequentialBlock();

        // Input Layer
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .setFilters(32)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);   
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2), new Shape(0, 0)));
        block.add(Dropout.builder().optRate(0.25f).build());

        // Second Layer
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .setFilters(64)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);   
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2), new Shape(0, 0)));
        block.add(Dropout.builder().optRate(0.25f).build());

        // Third Layer
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .setFilters(128)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);   
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2), new Shape(0, 0)));
        block.add(Dropout.builder().optRate(0.25f).build());

        // Flatten the output from the last pooling layer
        block.add(Blocks.batchFlattenBlock());

        // Dense Layer
        block.add(Linear.builder()
                .setUnits(512)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);   
        block.add(Dropout.builder().optRate(0.5f).build());

        // Output Layer
        block.add(Linear.builder()
                .setUnits(NUM_OF_OUTPUT)
                .build());
        block.add(new LambdaBlock(input -> new NDList(input.singletonOrThrow().softmax(1))));

        return block;
    }
}

/* 

// A helper class loads and saves model.
public final class Models {

    // Number of Classification-Labels
    public static final int NUM_OF_OUTPUT = 8;

    // Height and Width of Images
    public static final int IMAGE_HEIGHT = 128;
    public static final int IMAGE_WIDTH = 128;

    // Model-Name
    public static final String MODEL_NAME = "fruitclassifier";

    private Models() {}


    public static Model getModel() {

        // Create new instance of an empty model
        Model model = Model.newInstance(MODEL_NAME);

        // Create Model-Block
        Block modelBlock = getCustomModel();

        // Create Model with the created block
        model.setBlock(modelBlock);

        return model;
    }


    // Save model
    public static void saveSynset(Path modelDir, List<String> synset) throws IOException {
        Path synsetFile = modelDir.resolve("synset.txt");
        try (Writer writer = Files.newBufferedWriter(synsetFile)) {
            writer.write(String.join("\n", synset));
        }
    }

    // Create the Neural Network
    public static Block getCustomModel() {
        // Inspired by Python-Code:
        // https://www.kaggle.com/code/khushijoshi12344/fruit-dataset-image-classification-cnn/notebook

        SequentialBlock block = new SequentialBlock();

        // Input Layer
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))                                    // Size of the filters of the kernel is set to 3x3 pixels, so only a small part of the picture is processed at a time
                .optPadding(new Shape(1, 1))                                        // Output should have the same dimension as the input
                .setFilters(32)                                                      // Number of feature-maps which represent different aspects of the input image
                .build());
        block.add(BatchNorm.builder().build());                                              // Normalizes the batch, so there is no shifting of the parameters during the training
        block.add(Activation::relu);                                                         // Activation-Function
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));  // Reduces the dimension of the input for the next layer
        block.add(Dropout.builder().optRate((float) 0.25).build());                          // Droping out random units should prevent overfitting in neural networks

        // Second Layer
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .setFilters(64)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
        block.add(Dropout.builder().optRate((float) 0.25).build());

        // Third Layer
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .setFilters(128)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
        block.add(Dropout.builder().optRate((float) 0.25).build());

        block.add(Blocks.batchFlattenBlock());

        // Dense Layer
        block.add(Linear.builder()
                .setUnits(512)
                .build());
        block.add(BatchNorm.builder().build());
        block.add(Activation::relu);
        block.add(Dropout.builder().optRate((float) 0.5).build());

        // Output Layer
        block.add(Linear.builder()
                .setUnits(NUM_OF_OUTPUT)
                .build());
        block.add(new LambdaBlock(input -> new NDList(input.singletonOrThrow().softmax(1))));

        return block;
    }
    
}
*/
