# TorchSharpFlowerClassifier
Demo - TorchSharp image classification

This is a port of my [CSnakes-based image classifier](https://github.com/PhasonMatrix/CSnakesFlowerClassifier). This project has no python code, only C#. It uses [TorchSharp](https://github.com/dotnet/TorchSharp) for training and inference. At first I was hesitant to use TorchSharp as I had previously tried TensorFlow.Net and had a bad experience with bugs and the project was not maintained well. The developer experience with TorchSharp, however, is very good.

In comparison to the CSnakes project, I much prefer using TorchSharp directly in C# than using SCnakes and PyTorch. The code is smaller and cleaner. I get all the benefits of the C# language with no real downside. Everything that can be done in PyTorch can be done in TorchSharp, for this project anyway. I also don't need to a full python environment as part of the app. Inference time in CSnakes/PyTorch was around 150ms, in this project it averages around 50ms - 3x faster! That includes reading the image file and converting to a tensor. In the CSnakes project there was another step of tranfering the image pixel bytes from C# to Python.

<img width="1508" height="937" alt="TorchSharpFlowerClassifierInference" src="https://github.com/user-attachments/assets/a57b75b2-0477-47a0-84cd-a77744f75d4c" />

I added some progress bars for show the progress of the training process. This would have been a pain in CSnakes but was fairly simple using TorchSharp.

<img width="1513" height="947" alt="TorchSharpFlowerClassifierTraining" src="https://github.com/user-attachments/assets/f07a93ed-0f98-4d7c-9f7b-fe8121dded39" />



