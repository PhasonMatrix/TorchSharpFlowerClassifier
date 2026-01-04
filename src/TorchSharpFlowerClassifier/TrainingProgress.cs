using System;
using System.Collections.Generic;
using System.Text;

namespace TorchSharpFlowerClassifier;

public class TrainingProgress
{
    public double EpochCompletionPercentage { get; set; }
    public double TotalCompletionPercentage { get; set; }
    public string Status { get; set; }

}
