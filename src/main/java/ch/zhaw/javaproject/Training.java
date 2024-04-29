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
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * In training, multiple passes (or epochs) are made over the training data
 * trying to find patterns
 * and trends in the data, which are then stored in the model. During the
 * process, the model is
 * evaluated for accuracy using the validation data. The model is updated with
 * findings over each
 * epoch, which improves the accuracy of the model.
 */
public final class Training {

    // represents number of training samples processed before the model is updated
    private static final int BATCH_SIZE = 10;

    // the number of passes over the complete dataset
    private static final int EPOCHS = 8;

    public static void main(String[] args) throws IOException, TranslateException {
        // the location to save the model
        Path modelDir = Paths.get("models");
        
        // Dataset from https://www.kaggle.com/datasets/shreyapmaher/fruits-dataset-images/data
        ImageFolder dataset = initDataset("fruits");
        
        // Split the dataset set into training dataset and validate dataset
        RandomAccessDataset[] datasets = dataset.randomSplit(7, 3);

        RandomAccessDataset train = datasets[0];
        RandomAccessDataset test = datasets[1];

        // set loss function, which seeks to minimize errors
        // loss function evaluates model's predictions against the correct answer
        // (during training)
        // higher numbers are bad - means model performed poorly; indicates more errors;
        // want to
        // minimize errors (loss)
        Loss loss = Loss.softmaxCrossEntropyLoss();

        // setting training parameters (ie hyperparameters)
        TrainingConfig config = setupTrainingConfig(loss);

        Model model = Models.getModel(); // empty model instance to hold patterns
        Trainer trainer = model.newTrainer(config);
        // metrics collect and report key performance indicators, like accuracy
        trainer.setMetrics(new Metrics());

        Shape inputShape = new Shape(1, 3, Models.IMAGE_HEIGHT, Models.IMAGE_HEIGHT);

        // initialize trainer with proper input shape
        trainer.initialize(inputShape);

        // find the patterns in data
        EasyTrain.fit(trainer, EPOCHS, train, test);

        // set model properties
        TrainingResult result = trainer.getTrainingResult();
        model.setProperty("Epoch", String.valueOf(EPOCHS));
        model.setProperty(
                "Accuracy", String.format("%.5f", result.getValidateEvaluation("Accuracy")));
        model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));

        // save the model after done training for inference later
        // model saved as shoeclassifier-0000.params
        model.save(modelDir, Models.MODEL_NAME);

        // save labels into model directory
        Models.saveSynset(modelDir, dataset.getSynset());

    }

    private static ImageFolder initDataset(String datasetRoot)
            throws IOException, TranslateException {
        ImageFolder dataset = ImageFolder.builder()
                .setRepositoryPath(Paths.get(datasetRoot))                          // Retrieve the data
                .optMaxDepth(10)                                           // Number of Subfolders that are taken into consideration
                .addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT))  // Resize image for better performance
                .addTransform(new ToTensor())                                       // Convert images to tensor that can be used for Neural Networks
                .setSampling(BATCH_SIZE, true)                               // Number of Images processed at a time; Random-Sampling to process the data in random order
                .build();

        dataset.prepare();
        return dataset;
    }

    private static TrainingConfig setupTrainingConfig(Loss loss) {
        return new DefaultTrainingConfig(loss)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }
}
