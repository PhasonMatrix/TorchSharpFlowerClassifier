using Avalonia.Input;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using CommunityToolkit.Mvvm.ComponentModel;
using GUI.Models;
using GUI.Services;
using Microsoft.Extensions.DependencyInjection;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TorchSharpFlowerClassifier;

namespace GUI.ViewModels;

public partial class MainWindowViewModel : ViewModelBase
{

    private Inference _inference;
    private Training _training;


    public ObservableCollection<ResultsDisplayModel> DisplayResults { get; } = new();

    [ObservableProperty]
    private string _classificationMessage = "";

    public ObservableCollection<string> LogMessages { get; } = new ObservableCollection<string>();

    [ObservableProperty]
    private Cursor _cursor = new Cursor(StandardCursorType.Arrow);

    [ObservableProperty]
    private string _logText = string.Empty;

    [ObservableProperty]
    private double _epochProgress;

    [ObservableProperty]
    private double _totalProgress;

    [ObservableProperty]
    private Bitmap? _displayBitmap;


    [ObservableProperty]
    private string _modelFileName = "model_01.pth";


    [ObservableProperty]
    private string _trainingDataDir = "C:/Dev/ML DataSets/flower_photos";

    public event Action<TrainingResultsViewModel>? TrainingResultsReady;



    public MainWindowViewModel()
    {
        LogMessages.CollectionChanged += (s, e) => UpdateLogText();
    }

    private void UpdateLogText()
    {
        LogText = string.Join("\n", LogMessages);
    }



    public async void LoadAndClassifyButtonClick()
    {
        ClassificationMessage = "";
        Cursor = new Cursor(StandardCursorType.Wait);

        try
        {
            IFilesService? filesService = App.Current?.Services?.GetService<IFilesService>();
            if (filesService is null) { throw new NullReferenceException("Missing File Service instance."); }

            IStorageFile? file = await filesService.OpenFileAsync();
            if (file is null) { return; }

            await using var readStream = await file.OpenReadAsync();

            // Decode to SkiaSharp.SKBitmap
            SKBitmap skBitmap = SKBitmap.Decode(readStream);

            DisplayBitmap = SKBitmapToAvaloniaBitmap(skBitmap);

            string filePath = file.TryGetLocalPath();


            if (_inference == null)
            {
                LogMessages.Add($"Loading model...");
                _inference = new Inference();
            }


            // Run prediction on a background thread
            Dictionary<string, double> result = await Task.Run(() => _inference.PredictFromBitmap(skBitmap, ModelFileName));

            DisplayResults.Clear();
            foreach (var kv in result)
            {
                DisplayResults.Add(new ResultsDisplayModel()
                {
                    Classification = kv.Key,
                    Probability = (kv.Value * 100.0).ToString("0.00") + "%"
                });
            }

            KeyValuePair<string, double> topPair = result.OrderByDescending(kv => kv.Value).First();

            ClassificationMessage = $"Class: {topPair.Key}";
            LogMessages.Add($"Inference time: {_inference.LastInferenceDuration}");
            LogMessages.Add($"Inference result: {string.Join(",", result.Select(kv => kv.Key + ":" + kv.Value.ToString("0.000")).ToArray())}");
        }
        catch (Exception ex)
        {
            LogMessages.Add($"Error: {ex.Message}");
        }
        finally
        {
            Cursor = new Cursor(StandardCursorType.Arrow);
        }
    }





    public async void TrainModelButtonClick()
    {

        ClassificationMessage = "";
        Cursor = new Cursor(StandardCursorType.Wait);
        try
        {
            if (_training == null)
            {
                var progress = new Progress<TrainingProgress>(progress =>
                {
                    // Update the log messages based on progress updates
                    LogMessages.Add($"{progress.Status}");
                    EpochProgress = progress.EpochCompletionPercentage;
                    TotalProgress = progress.TotalCompletionPercentage;
                });

                _training = new Training(progress);
            }



            await Task.Run(() => _training.TrainAndSaveModel(TrainingDataDir, ModelFileName));
            LogMessages.Add("Training completed successfully.");

            // display training results
            var resultsVm = new TrainingResultsViewModel();
            resultsVm.LoadTrainingResults(
                _training?.TrainingAccuracies ?? new List<double>(),
                _training?.ValidationAccuracies ?? new List<double>(),
                _training?.TrainingLosses ?? new List<double>(),
                _training?.ValidationLosses ?? new List<double>()
            );

            // Raise the event
            TrainingResultsReady?.Invoke(resultsVm);

        }
        catch (Exception ex)
        {
            LogMessages.Add($"Error: {ex.Message}");
        }
        finally
        {
            Cursor = new Cursor(StandardCursorType.Arrow);
        }
    }



    public Bitmap SKBitmapToAvaloniaBitmap(SKBitmap skBitmap)
    {
        SKData data = skBitmap.Encode(SKEncodedImageFormat.Png, 100);
        using (Stream stream = data.AsStream())
        {
            return new Bitmap(stream);
        }
    }


}
