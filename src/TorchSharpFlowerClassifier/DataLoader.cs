
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using SkiaSharp;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions.transforms;

namespace TorchSharpFlowerClassifier;

internal class DataLoader : IEnumerable<(Tensor image, Tensor label)> //, IDisposable
{
    private List<(string path, long label)> _samples = new();
    private readonly int _imageSize;

    public bool ShuffleSamples { get; set; }

    public Dictionary<string, long> ClassToIndex { get; } = new();
    public int ClassCount => ClassToIndex.Count;
    public int Count => _samples.Count;


    public DataLoader(string rootDir, int imageSize = 256)
    {
        _imageSize = imageSize;

        // Discover class folders
        var classDirs = Directory.GetDirectories(rootDir);
        Array.Sort(classDirs, StringComparer.Ordinal);

        for (int i = 0; i < classDirs.Length; i++)
        {
            ClassToIndex[Path.GetFileName(classDirs[i])] = i;
        }


        // Load image paths + labels
        foreach (var classDir in classDirs)
        {
            var className = Path.GetFileName(classDir);
            var label = ClassToIndex[className];

            foreach (var file in Directory.EnumerateFiles(classDir))
            {
                if (IsImageFile(file))
                {
                    _samples.Add((file, label));
                }
            }
        }

    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();


    public IEnumerator<(Tensor image, Tensor label)> GetEnumerator()
    {
        // reproducible shuffle
        var rnd = new Random(42);
        _samples = ShuffleSamples ? _samples.OrderBy(_ => rnd.Next()).ToList() : _samples;

        foreach (var (path, label) in _samples)
        {

            SKBitmap bitmap = SKBitmap.Decode(path);

            SKSamplingOptions samplingOptions;
            if (_imageSize > bitmap.Width || _imageSize > bitmap.Height)
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
            SKBitmap resizedBitmap = bitmap.Resize(new SKImageInfo(_imageSize, _imageSize, SKColorType.Rgba8888), samplingOptions);


            var imageTensor = ImageToTensor(resizedBitmap);
            var labelTensor = torch.tensor(label, dtype: ScalarType.Int64);

            bitmap.Dispose();
            resizedBitmap.Dispose();

            yield return (imageTensor, labelTensor);
        }
    }



    private static bool IsImageFile(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
    }


    public static Tensor ImageToTensor(SKBitmap bitmap)
    {
        // Ensure the bitmap is in a compatible color format (RGBA 8888 is standard)
        if (bitmap.ColorType != SKColorType.Rgba8888)
        {
            // Convert if necessary (creating a new bitmap)
            SKBitmap convertedBitmap = new SKBitmap(bitmap.Width, bitmap.Height, SKColorType.Rgba8888, SKAlphaType.Premul);
            bitmap.CopyTo(convertedBitmap, SKColorType.Rgba8888);
            bitmap = convertedBitmap;
        }

        int width = bitmap.Width;
        int height = bitmap.Height;


        // Create a managed byte array copy
        byte[] imageData = bitmap.Bytes;


        // Define the tensor shape: (Height, Width, Channels) 
        // For RGBA 8888, there are 4 channels.
        long[] dimensions = { height, width, 4 };

        // Create the TorchSharp tensor from the managed array
        // We use byte (unsigned 8-bit integer) as the data type (ScalarType.Byte)
        // Torch.from_array avoids copying data if possible, but ToArray() above made a copy anyway.
        // Torch.tensor will work as well.
        Tensor tensor = torch.tensor(imageData, dtype: ScalarType.Byte)
                             .reshape(dimensions);

        // Scale from integer 0-255 range to float 0.0-1.0 range
        // channels first (Channels, Height, Width).
        Tensor floatTensor = tensor.to(ScalarType.Float32)
                                   .div(255.0f); 

        // Optional: Permute dimensions from HWC to CHW (standard PyTorch format)
        Tensor inputTensor = floatTensor.permute(2, 0, 1);

        // Drop alpha channel: keep only RGB
        inputTensor = inputTensor.slice(0, 0, 3, 1);


        // Remember to dispose of tensors when you are done with them to free unmanaged memory
        tensor.Dispose();
        floatTensor.Dispose();

        return inputTensor;
    }

}
