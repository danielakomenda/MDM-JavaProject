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
import ai.djl.modality.cv.transform.RandomFlipLeftRight;
import ai.djl.modality.cv.transform.RandomResizedCrop;
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
import java.net.URI;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class Training {

    private static final int BATCH_SIZE = 64; // Smaller Batch-Size, because the dataset is quite small
    private static final int EPOCHS = 1;

    public static void main(String[] args) throws IOException, TranslateException {
        // The location to save the model
        Path modelDir = Paths.get("models");

        // Dataset from
        // https://www.kaggle.com/datasets/shreyapmaher/fruits-dataset-images/data
        ImageFolder dataset = initDataset("fruits");

        // Split the dataset set into training dataset and validate dataset
        RandomAccessDataset[] datasets = dataset.randomSplit(7, 3);

        RandomAccessDataset train = datasets[0];
        RandomAccessDataset test = datasets[1];

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

        // Save Model after the training as fruitclassifier-<numberOfEpochs>.params
        // Save Classification-Labels as synset.txt
        model.save(modelDir, Models.MODEL_NAME);
        Models.saveSynset(modelDir, dataset.getSynset());

        List<String> synset = dataset.getSynset();

        addFileToZip(modelDir, Models.MODEL_NAME, synset);

    }

    private static ImageFolder initDataset(String datasetRoot)
            throws IOException, TranslateException {
        ImageFolder dataset = ImageFolder.builder()
                .setRepositoryPath(Paths.get(datasetRoot)) // Retrieve the data
                .optMaxDepth(10) // Number of Subfolders that are taken into consideration
                .addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT)) // Resize image for better
                                                                                   // performance
                .addTransform(new RandomFlipLeftRight()) // Data-Augmentation: Random horizontal flip
                .addTransform(new RandomResizedCrop( // Data-Augmentation: Random crop and resize
                        Models.IMAGE_WIDTH,
                        Models.IMAGE_HEIGHT,
                        0.8f, 0, 0, 0))
                .addTransform(new ToTensor()) // Convert images to tensor that can be used for Neural Networks
                .setSampling(BATCH_SIZE, true) // Number of Images processed at a time; Random-Sampling to process the
                                               // data in random order
                .build();

        dataset.prepare();
        return dataset;
    }

    private static TrainingConfig setupTrainingConfig(Loss loss) {
        return new DefaultTrainingConfig(loss)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }


    private static void addFileToZip(Path modelDir, String model, List<String> synset) throws IOException {
        String modelFileName = model + "-" + String.format("%04d", EPOCHS) + ".params";
        Path synsetFile = modelDir.resolve("synset.txt");
        URI uri = URI.create("jar:file:" + modelDir.resolve("fruitclassifier.zip").toUri().getPath());
    
        Map<String, String> env = new HashMap<>();
        env.put("create", "true");
    
        // Printing for debugging
        System.out.println("Model Directory: " + modelDir);
        System.out.println("Model File Name: " + modelFileName);
        System.out.println("Synset File: " + synsetFile);
        System.out.println("URI: " + uri);
    
        try (FileSystem zipfs = FileSystems.newFileSystem(uri, env)) {
            Path modelFile = modelDir.resolve(modelFileName);
            Path modelInZip = zipfs.getPath(modelFileName);  // Only filename, no absolute path
            Path synsetInZip = zipfs.getPath("synset.txt");  // Only filename, no absolute path
    
            // Copy a file into the zip file
            Files.copy(modelFile, modelInZip, StandardCopyOption.REPLACE_EXISTING);
            Files.copy(synsetFile, synsetInZip, StandardCopyOption.REPLACE_EXISTING);
        }
    }
    
}
