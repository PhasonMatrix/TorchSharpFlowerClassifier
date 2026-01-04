using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpFlowerClassifier;

internal class ClassifierModel : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _features;
    private readonly Module<Tensor, Tensor> _classifier;


    public ClassifierModel(int numClasses = 5, torch.Device? device = null) : base(nameof(ClassifierModel))
    {
        // Convolutional feature extractor
        _features = Sequential(
            ("conv1", Conv2d(3, 16, kernel_size: 3, stride: 1, padding: 1)),
            ("relu1", ReLU()),
            ("pool1", MaxPool2d(2)),
            ("drop1", Dropout2d(0.4)),

            ("conv2", Conv2d(16, 32, kernel_size: 3, stride: 1, padding: 1)),
            ("relu2", ReLU()),
            ("pool2", MaxPool2d(2)),
            ("drop2", Dropout2d(0.4)),

            ("conv3", Conv2d(32, 64, kernel_size: 3, stride: 1, padding: 1)),
            ("relu3", ReLU()),
            ("pool3", MaxPool2d(2)),
            ("drop3", Dropout2d(0.4)),

            ("conv4", Conv2d(64, 128, kernel_size: 3, stride: 1, padding: 1)),
            ("relu4", ReLU()),
            ("pool4", MaxPool2d(2))
        );

        // Classifier head
        _classifier = Sequential(
            ("fc1", Linear(128 * 16 * 16, 128)),
            ("relu5", ReLU()),
            ("drop4", Dropout(0.5)),

            ("fc2", Linear(128, 64)),
            ("relu6", ReLU()),
            ("drop5", Dropout(0.5)),

            ("fc3", Linear(64, numClasses))
        );

        RegisterComponents();
        if (device != null && device.type != DeviceType.CPU)
        {
            this.to(device);
        }
            
    }


    public override Tensor forward(Tensor x)
    {
        x = _features.forward(x);
        x = x.view(x.shape[0], -1); // flatten
        x = _classifier.forward(x);
        return x;
    }
}

