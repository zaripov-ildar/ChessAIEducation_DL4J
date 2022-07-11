import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.IOException;

public class Tester {
    public static void main(String[] args) throws IOException {
        File modelFile = new File("src/main/resources/model.zip");

        MultiLayerNetwork model;
        if (modelFile.exists()) {
            model = ModelSerializer.restoreMultiLayerNetwork("src/main/resources/model.zip", true);
            DataSet testSet = Trainer.getData("d://dataDirectory//1");
            INDArray output = model.output(testSet.getFeatureMatrix());
            Evaluation eval = new Evaluation(Trainer.layers[Trainer.layers.length-1]);
            eval.eval(testSet.getLabels(), output);
            System.out.println(eval.stats());
        }
    }
}
