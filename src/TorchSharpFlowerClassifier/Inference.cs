using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Text;
using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace TorchSharpFlowerClassifier;

public class Inference
{

    private ClassifierModel _model;
    private string[] _classIndexToName =
    {
        "daisy",
        "dandelion",
        "roses",
        "sunflowers",
        "tulips"
    };

    public TimeSpan LastInferenceDuration { get; private set; } = TimeSpan.Zero;



    public Dictionary<string, double> PredictFromBitmap(SKBitmap bitmap, string modelFileName)
    {
        DateTime startTime = DateTime.Now;

        SKSamplingOptions samplingOptions;
        if (Training.ImageSize > bitmap.Width || Training.ImageSize > bitmap.Height)
        {
            // Upscaling: Using Mitchell cubic resampler for good quality
            samplingOptions = new SKSamplingOptions(SKCubicResampler.Mitchell);
        }
        else
        {
            // Downscaling: Using linear filtering with linear mipmaps for anti-aliasing
            samplingOptions = new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear);
        }
        // resize
        SKBitmap resizedBitmap = bitmap.Resize(new SKImageInfo(Training.ImageSize, Training.ImageSize, SKColorType.Rgba8888), samplingOptions);

        var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;


        Tensor imageTensor = DataLoader.ImageToTensor(resizedBitmap).unsqueeze(0); // Add batch dimension
        imageTensor = imageTensor.to(device);

        if (_model == null)
        {
            LoadModel(modelFileName, device);
        }


        Dictionary<string, double> result = new();

        using (torch.no_grad())
        {
            // Forward pass
            var outputs = _model.forward(imageTensor);

            // Softmax over class dimension
            var probabilities = outputs.softmax(1);

            // Remove batch dimension: [1, C] -> [C]
            var probs = probabilities.squeeze(0);

            // Convert to managed array
            float[] probArray = probs.cpu().data<float>().ToArray();

            // Map to class names
            for (int i = 0; i < probArray.Length; i++)
            {
                string className = _classIndexToName[i];
                result[className] = probArray[i];
            }

            // Cleanup
            outputs.Dispose();
            probabilities.Dispose();
            probs.Dispose();
        }

        imageTensor.Dispose();

        LastInferenceDuration = DateTime.Now - startTime;

        return result;
    }



    private void LoadModel(string modelFileName, torch.Device device)
    {
        string modelPath = Path.Combine(Training.ModelWeightsDirectory, modelFileName);
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model file not found: {modelPath}");
        }

        _model = new ClassifierModel(5);
        _model.to(device);
        _model.load(modelPath);
        _model.eval();
    }

}
