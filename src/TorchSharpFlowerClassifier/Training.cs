using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;

namespace TorchSharpFlowerClassifier;

public class Training
{
    private readonly IProgress<TrainingProgress>? _progressReporter;

    public static string ModelWeightsDirectory => "ModelWeights";
    public static int ImageSize => 256;

    private int _epochs = 80;
    private int _batchSize = 64;

    // metrics
    public List<double> TrainingAccuracies { get; } = new();
    public List<double> TrainingLosses { get; } = new();
    public List<double> ValidationAccuracies { get; } = new();
    public List<double> ValidationLosses { get; } = new();



    public Training(IProgress<TrainingProgress>? progressReporter = null)
    {
        _progressReporter = progressReporter;
    }


    public void TrainAndSaveModel(string trainingDataDirectory, string modelFileName)
    {
        TrainingAccuracies.Clear();
        TrainingLosses.Clear();
        ValidationAccuracies.Clear();
        ValidationLosses.Clear();


        string modelSavePath = Path.Combine(ModelWeightsDirectory, modelFileName);
        // make folder if it doesn't exist
        if (!Directory.Exists(ModelWeightsDirectory))
        {
            Directory.CreateDirectory(ModelWeightsDirectory);
        }

        // use GPU if available, otherwise CPU
        torch.Device device = torch.CPU;
        if (torch.cuda.is_available())
        {
            device = torch.CUDA;
            Debug.WriteLine("CUDA is available. Using GPU for training.");
            _progressReporter?.Report(new TrainingProgress
            {
                EpochCompletionPercentage = 0,
                TotalCompletionPercentage = 0,
                Status = $"CUDA is available. Using GPU for training."
            });
        }
        else
        {
            Debug.WriteLine("CUDA is not available. Using CPU for training.");
            _progressReporter?.Report(new TrainingProgress
            {
                EpochCompletionPercentage = 0,
                TotalCompletionPercentage = 0,
                Status = $"CUDA is not available. Using CPU for training."
            });
        }



        var resize   = transforms.Resize(ImageSize, ImageSize);
        var hflip    = transforms.HorizontalFlip();
        var rotate   = transforms.Rotate(10);
        var color    = transforms.ColorJitter(brightness: 0.2f, contrast: 0.2f, saturation: 0.2f, hue: 0.1f);
        var contrast = transforms.AdjustContrast(1.2f);
        
        var transform = new ITransform[]
        {
            resize,
            hflip,
            rotate,
            color,
            contrast
        };


        DataLoader dataLoader = new DataLoader(trainingDataDirectory, ImageSize);
        dataLoader.ShuffleSamples = true;
        Debug.WriteLine($"Found {dataLoader.Count} images belonging to {dataLoader.ClassCount} classes.");
        foreach (KeyValuePair<string, long> kvp in dataLoader.ClassToIndex)
        {
            Debug.WriteLine($"Class: {kvp.Key}, Index: {kvp.Value}");
            _progressReporter?.Report(new TrainingProgress
            {
                EpochCompletionPercentage = 0,
                TotalCompletionPercentage = 0,
                Status = $"Class: {kvp.Key}, Index: {kvp.Value}"
            });
        }


        // calculate split sizes
        int trainSize = (int)(dataLoader.Count * 0.8);
        int validationSize = (int)(dataLoader.Count * 0.15);
        int testSize = dataLoader.Count - trainSize - validationSize;

        IEnumerable<(Tensor image, Tensor label)> trainSamples = dataLoader.Take(trainSize);
        IEnumerable<(Tensor image, Tensor label)> valSamples = dataLoader.Skip(trainSize).Take(validationSize);
        IEnumerable<(Tensor image, Tensor label)> testSamples = dataLoader.Skip(trainSize + validationSize);


        var model = new ClassifierModel(numClasses: dataLoader.ClassCount);
        model.to(device);

        var lossFn = torch.nn.CrossEntropyLoss();

        var optimizer = torch.optim.Adam(
            model.parameters(),
            lr: 0.0002,
            weight_decay: 1e-5
        );

        var trainingStopwatch = Stopwatch.StartNew();


        // epoch loop
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            model.train();

            double trainLoss = 0;
            long trainCorrect = 0;
            long trainTotal = 0;

            _progressReporter?.Report(new TrainingProgress
            {
                EpochCompletionPercentage = 0,
                TotalCompletionPercentage = (double)epoch / (double)_epochs * 100.0,
                Status = $"Loading training data for epoch {epoch + 1}/{_epochs}..."
            });

            var trainBatches = Batch(trainSamples, _batchSize, device, shuffle: true).ToList();
            int totalBatches = trainBatches.Count;

            // batch loop
            for (int batchIndex = 0; batchIndex < totalBatches; batchIndex++)
            {
                var (X, y) = trainBatches[batchIndex];
                //Debug.WriteLine(string.Join(", ", X.shape));
                var yPred = model.forward(X);
                var loss = lossFn.forward(yPred, y);

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                trainLoss += loss.ToSingle();

                var predicted = yPred.argmax(1);
                trainTotal += y.shape[0];
                trainCorrect += predicted.eq(y).sum().ToInt64();
                Debug.WriteLine($"  Training: {batchIndex + 1}/{totalBatches} batches | Loss: {loss.ToSingle():F4}");
                _progressReporter?.Report(new TrainingProgress
                {
                    EpochCompletionPercentage = (double)batchIndex / (double)totalBatches * 100.0,
                    TotalCompletionPercentage = (double)epoch / (double)_epochs * 100.0,
                    Status = $"  Training: {batchIndex + 1}/{totalBatches} batches | Loss: {loss.ToSingle():F4}"
                });


                predicted.Dispose();
                loss.Dispose();
            }

            // ---------------- validation for this epoch ----------------

            model.eval();

            double valLoss = 0;
            long valCorrect = 0;
            long valTotal = 0;

            _progressReporter?.Report(new TrainingProgress
            {
                EpochCompletionPercentage = 0,
                TotalCompletionPercentage = (double)epoch / (double)_epochs * 100.0,
                Status = $"Loading validation data for epoch {epoch + 1}/{_epochs}..."
            });
            var valBatches = Batch(valSamples, _batchSize, device).ToList();
            int totalValBatches = valBatches.Count;

            using (torch.no_grad())
            {
                for (int batchIndex = 0; batchIndex < totalValBatches; batchIndex++)
                {
                    var (X, y) = valBatches[batchIndex];
                    var yPred = model.forward(X);
                    var loss = lossFn.forward(yPred, y);

                    valLoss += loss.ToSingle();

                    var predicted = yPred.argmax(1);
                    valTotal += y.shape[0];
                    valCorrect += predicted.eq(y).sum().ToInt64();

                    Debug.WriteLine($"  Validation: {batchIndex + 1}/{totalValBatches} batches");
                    _progressReporter?.Report(new TrainingProgress
                    {
                        EpochCompletionPercentage = (double)batchIndex / (double)totalValBatches * 100.0,
                        TotalCompletionPercentage = (double)epoch / (double)_epochs * 100.0,
                        Status = $"  Validation: {batchIndex + 1}/{totalValBatches} batches"
                    });

                    predicted.Dispose();
                    loss.Dispose();
                }
            }

            double avgTrainLoss = trainLoss / trainBatches.Count;
            double avgValLoss = valLoss / totalValBatches;

            double trainAccuracy = 100.0 * trainCorrect / trainTotal;
            double valAccuracy = 100.0 * valCorrect / valTotal;

            TrainingLosses.Add(avgTrainLoss);
            ValidationLosses.Add(avgValLoss);
            TrainingAccuracies.Add(trainAccuracy);
            ValidationAccuracies.Add(valAccuracy);


            // ETA calculation
            double elapsedSeconds = trainingStopwatch.Elapsed.TotalSeconds;
            double avgEpochSeconds = elapsedSeconds / (epoch + 1);
            int remainingEpochs = _epochs - (epoch + 1);
            double etaSeconds = remainingEpochs * avgEpochSeconds;

            int etaMinutes = (int)(etaSeconds / 60);
            int etaHours = etaMinutes / 60;
            etaMinutes %= 60;

            var eta = TimeSpan.FromSeconds(etaSeconds);

            string updateMessage = 
                $"{DateTime.Now.ToString("HH:mm:ss")} | " + 
                $"Epoch {epoch + 1}/{_epochs} | " +
                $"Train Loss: {avgTrainLoss:F4} | Val Loss: {avgValLoss:F4} | " +
                $"Train Acc: {trainAccuracy:F2}% | Val Acc: {valAccuracy:F2}% | " +
                $"ETA: {eta:hh\\:mm\\:ss} | Avg epoch: {avgEpochSeconds:F1}s"
            ;

            Debug.WriteLine(updateMessage);
            _progressReporter?.Report(new TrainingProgress
            {
                EpochCompletionPercentage = 0.0,
                TotalCompletionPercentage = (double)epoch / (double)_epochs * 100.0,
                Status = updateMessage
            });

        }

        // test 
        model.eval();

        double testLoss = 0;
        long testCorrect = 0;
        long testTotal = 0;

        using (torch.no_grad())
        {
            foreach (var (X, y) in Batch(testSamples, _batchSize, device))
            {
                var yPred = model.forward(X);
                var loss = lossFn.forward(yPred, y);

                testLoss += loss.ToSingle();

                var predicted = yPred.argmax(1);
                testTotal += y.shape[0];
                testCorrect += predicted.eq(y).sum().ToInt64();

                predicted.Dispose();
                loss.Dispose();
            }
        }

        double testAccuracy = 100.0 * testCorrect / testTotal;
        double avgTestLoss = testLoss / Math.Max(1, testSamples.Count());

        Debug.WriteLine($"Test Loss: {avgTestLoss:F4}");
        Debug.WriteLine($"Test Accuracy: {testAccuracy:F2}%");
        
        
        // save the model
        model.save(modelSavePath);
    }




    private static IEnumerable<(Tensor X, Tensor y)> Batch(
        IEnumerable<(Tensor image, Tensor label)> data,
        int batchSize,
        Device device,
        bool shuffle = false)
    {
        var enumerator = shuffle ? data.OrderBy(_ => Guid.NewGuid()).GetEnumerator() : data.GetEnumerator();

        while (true)
        {
            var batchImages = new List<Tensor>();
            var batchLabels = new List<Tensor>();

            for (int i = 0; i < batchSize && enumerator.MoveNext(); i++)
            {
                var (image, label) = enumerator.Current;
                batchImages.Add(image);
                batchLabels.Add(label);
            }

            if (batchImages.Count == 0) yield break;

            yield return (
                torch.stack(batchImages.ToArray()).to(device),
                torch.stack(batchLabels.ToArray()).to(device)
            );
        }
    }



}
