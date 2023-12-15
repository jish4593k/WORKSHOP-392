import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.LambdaLayer;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.nd4j.linalg.schedule.TbpttSchedule;

public class GANModelJava {

    public static class Generator {
        public static ComputationGraphConfiguration.GraphBuilder createGeneratorConfig(int noiseSize, int hiddenSize, int maxTrajLen) {
            return new NeuralNetConfiguration.Builder()
                    .seed(123)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(org.deeplearning4j.nn.conf.Updater.ADAM)
                    .weightInit(WeightInit.XAVIER)
                    .graphBuilder()
                    .addInputs("noiseInput")
                    .addLayer("layer1", new DenseLayer.Builder()
                            .nIn(noiseSize)
                            .nOut(64)
                            .activation(Activation.LEAKYRELU)
                            .build(), "noiseInput")
                    .addLayer("layer2", new DenseLayer.Builder()
                            .nIn(64)
                            .nOut(128)
                            .activation(Activation.LEAKYRELU)
                            .build(), "layer1")
                    .addLayer("layer3", new DenseLayer.Builder()
                            .nIn(128)
                            .nOut(maxTrajLen)
                            .activation(Activation.LEAKYRELU)
                            .build(), "layer2")
                    .addLayer("lambda", new LambdaLayer.Builder().lambda("X -> X.unsqueeze(1)").build(), "layer3")
                    .addLayer("batchNorm", new BatchNormalization.Builder().nIn(1).nOut(1).build(), "lambda")
                    .addLayer("conv1d", new Convolution1DLayer.Builder()
                            .nIn(1)
                            .nOut(3)
                            .kernelSize(7)
                            .padding(3)
                            .activation(Activation.TANH)
                            .build(), "batchNorm")
                    .setOutputs("conv1d");
        }
    }

    public static class Discriminator {
        public static ComputationGraphConfiguration.GraphBuilder createDiscriminatorConfig(int arrayLength, int hiddenSize) {
            return new NeuralNetConfiguration.Builder()
                    .seed(123)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(org.deeplearning4j.nn.conf.Updater.ADAM)
                    .weightInit(WeightInit.XAVIER)
                    .graphBuilder()
                    .addInputs("input")
                    .addLayer("lambda", new LambdaLayer.Builder().lambda("X -> X.squeeze(1)").build(), "input")
                    .addLayer("conv1d", new Convolution1DLayer.Builder()
                            .nIn(3)
                            .nOut(1)
                            .kernelSize(7)
                            .padding(3)
                            .stride(2)
                            .activation(Activation.LEAKYRELU)
                            .build(), "lambda")
                    .addLayer("lambda2", new LambdaLayer.Builder().lambda("X -> X.squeeze(1)").build(), "conv1d")
                    .addLayer("dense1", new DenseLayer.Builder()
                            .nIn(arrayLength / 2)
                            .nOut(64)
                            .activation(Activation.LEAKYRELU)
                            .build(), "lambda2")
                    .addLayer("dense2", new DenseLayer.Builder()
                            .nIn(64)
                            .nOut(32)
                            .activation(Activation.LEAKYRELU)
                            .build(), "dense1")
                    .addLayer("dense3", new DenseLayer.Builder()
                            .nIn(32)
                            .nOut(1)
                            .activation(Activation.LEAKYRELU)
                            .build(), "dense2")
                    .setOutputs("dense3");
        }
    }

    public static void main(String[] args) {
        int noiseSize = 32;
        int hiddenSize = 64;
        int maxTrajLen = 128;

        ComputationGraphConfiguration generatorConfig = Generator.createGeneratorConfig(noiseSize, hiddenSize, maxTrajLen);
        ComputationGraph generator = new org.deeplearning4j.nn.graph.ComputationGraph(generatorConfig);
        generator.init();

        int arrayLength = 128;
        ComputationGraphConfiguration discriminatorConfig = Discriminator.createDiscriminatorConfig(arrayLength, hiddenSize);
        ComputationGraph discriminator = new org.deeplearning4j.nn.graph.ComputationGraph(discriminatorConfig);
        discriminator.init();

        // Training code goes here
    }
}
