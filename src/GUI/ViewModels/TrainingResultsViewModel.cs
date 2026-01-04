using Avalonia.Media;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using SkiaSharp;
using System;
using System.Collections.Generic;


namespace GUI.ViewModels;


public partial class TrainingResultsViewModel : ViewModelBase
{

    private SKColor _trainingAccuracyColour = new SKColor(50, 200, 255); // Blue/Cyan
    private SKColor _validationAccuracyColour = new SKColor(50, 255, 100); // green
    private SKColor _trainingLossColour = new SKColor(255, 80, 80); // red/pink
    private SKColor _validationLossColour = new SKColor(255, 220, 50); // yellow/orange


    // plural of series is series. This is an array of many series
    public ISeries[] AccuracyDataSeries { get; set; }
    public ISeries[] LossDataSeries { get; set; }


    public IBrush TrainingAccuracyBrush => new SolidColorBrush(new Avalonia.Media.Color(
        _trainingAccuracyColour.Alpha,
        _trainingAccuracyColour.Red,
        _trainingAccuracyColour.Green,
        _trainingAccuracyColour.Blue));

    public IBrush ValidationAccuracyBrush => new SolidColorBrush(new Avalonia.Media.Color(
        _validationAccuracyColour.Alpha,
        _validationAccuracyColour.Red,
        _validationAccuracyColour.Green,
        _validationAccuracyColour.Blue));

    public IBrush TrainingLossBrush => new SolidColorBrush(new Avalonia.Media.Color(
        _trainingLossColour.Alpha,
        _trainingLossColour.Red,
        _trainingLossColour.Green,
        _trainingLossColour.Blue));

    public IBrush ValidationLossBrush => new SolidColorBrush(new Avalonia.Media.Color(
        _validationLossColour.Alpha,
        _validationLossColour.Red,
        _validationLossColour.Green,
        _validationLossColour.Blue));



    public void LoadTrainingResults(
        IReadOnlyList<double> trainingAccuracies,
        IReadOnlyList<double> validationAccuracies,
        IReadOnlyList<double> trainingLosses,
        IReadOnlyList<double> validationLosses)
    {
        AccuracyDataSeries = new ISeries[]
        {
            new LineSeries<double>
            {
                Values = trainingAccuracies,
                Name = "Training Accuracy",
                Stroke = new SolidColorPaint(_trainingAccuracyColour) { StrokeThickness = 2 },
                Fill = null,
                GeometrySize = 10,
                GeometryStroke = new SolidColorPaint(_trainingAccuracyColour) { StrokeThickness = 2 },
                GeometryFill = null
            },
            new LineSeries<double>
            {
                Values = validationAccuracies,
                Name = "Validation Accuracy",
                Stroke = new SolidColorPaint(_validationAccuracyColour) { StrokeThickness = 2 },
                Fill = null,
                GeometrySize = 10,
                GeometryStroke = new SolidColorPaint(_validationAccuracyColour) { StrokeThickness = 2 },
                GeometryFill = null
            },
        };

        LossDataSeries = new ISeries[]
        {
            new LineSeries<double>
            {
                Values = trainingLosses,
                Name = "Training Loss",
                Stroke = new SolidColorPaint(_trainingLossColour) { StrokeThickness = 2 },
                Fill = null,
                GeometrySize = 10,
                GeometryStroke = new SolidColorPaint(_trainingLossColour) { StrokeThickness = 2 },
                GeometryFill = null
            },
            new LineSeries<double>
            {
                Values = validationLosses,
                Name = "Validation Loss",
                Stroke = new SolidColorPaint(_validationLossColour) { StrokeThickness = 2 },
                Fill = null,
                GeometrySize = 10,
                GeometryStroke = new SolidColorPaint(_validationLossColour) { StrokeThickness = 2 },
                GeometryFill = null
            },
        };
    }

}