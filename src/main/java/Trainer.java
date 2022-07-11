import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.standalone.ClassPathResource;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Trainer {
    private static final int BATCH_SIZE = 1000;
    public static final int[] layers = new int[]{69, 150, 280, 560, 1320};
    private static final int LABEL_INDEX = layers[0];
    private static final int NUM_POSSIBLE_LABELS = layers[4] + LABEL_INDEX;

    public static void main(String[] args) {
        MultiLayerNetwork model = new MultiLayerNetwork(getConfig());
        model.init();
        List<File> fileList = getFileList();

        fileList.forEach(file -> {
            if (file.isFile()) {
                DataSet dataSet = getData(file.getPath());
                model.fit(dataSet);
                saveModel(model);
            }
        });
    }

    private static void saveModel(MultiLayerNetwork model) {
        try {
            ModelSerializer.writeModel(model, "src/main/resources/model.zip", true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static List<File> getFileList() {
        List<File> fileList = new ArrayList<>();
        File file = new File("d://dataDirectory//");
        File[] files = file.listFiles();
        if (files != null) {
            fileList = Arrays.asList(files);
        }
        return fileList;
    }

    public static DataSet getData(String path) {
        DataSet allData = null;
        try (RecordReader recordReader = new CSVRecordReader(0, ',')) {
//            data initialisation
            recordReader.initialize(new FileSplit(
                    new ClassPathResource(path).getFile()));
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, BATCH_SIZE, LABEL_INDEX, NUM_POSSIBLE_LABELS);

//            random order
            allData = iterator.next();
            allData.shuffle(System.currentTimeMillis());

//             normalization
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(allData);
            normalizer.transform(allData);

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return allData;
    }

    private static MultiLayerConfiguration getConfig() {
        return new NeuralNetConfiguration.Builder()
                .iterations(10)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)

                .regularization(true)
                .useDropConnect(true)
                .dropOut(0.5)

//                    .updater(new Nesterovs(learningRate, 0.5))
                .updater(new AdaGrad())
//                .updater(new Adam())
                .list()
                .layer(0, new DenseLayer.Builder()
                        .dropOut(0.5)
                        .nIn(layers[0])
                        .nOut(layers[1])
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .dropOut(0.5)
                        .nIn(layers[1])
                        .nOut(layers[2])
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.TANH)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .dropOut(0.5)
                        .nIn(layers[2])
                        .nOut(layers[3])
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.TANH)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .dropOut(0.5)
                        .weightInit(WeightInit.RELU)
                        .activation(Activation.IDENTITY)
                        .nIn(layers[3])
                        .nOut(layers[4])
                        .build())
                .pretrain(false).backprop(true).build();

    }
}
