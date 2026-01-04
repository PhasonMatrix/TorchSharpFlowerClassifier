using Avalonia.Controls;
using Avalonia.Threading;
using GUI.ViewModels;

namespace GUI.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        this.DataContextChanged += MainWindow_DataContextChanged;

        if (DataContext is ViewModels.MainWindowViewModel vm)
        {
            vm.TrainingResultsReady += ShowTrainingResultsDialog;
        }
    }




    private void MainWindow_DataContextChanged(object? sender, System.EventArgs e)
    {
        var logTextBox = this.FindControl<TextBox>("LogTextBox");
        if (logTextBox != null && DataContext is ViewModels.MainWindowViewModel vm)
        {
            vm.PropertyChanged += (s, args) =>
            {
                if (args.PropertyName == nameof(vm.LogText))
                {
                    // Scroll to end when LogText changes
                    Dispatcher.UIThread.Post(() =>
                    {
                        logTextBox.CaretIndex = logTextBox.Text?.Length ?? 0;
                        logTextBox.ScrollToLine(logTextBox.GetLineCount() - 1);
                    });
                }
            };

            vm.TrainingResultsReady -= ShowTrainingResultsDialog; // avoid double subscription
            vm.TrainingResultsReady += ShowTrainingResultsDialog;
        }
    }


    private async void ShowTrainingResultsDialog(TrainingResultsViewModel resultsVm)
    {
        var resultsWindow = new TrainingResultsWindow
        {
            DataContext = resultsVm
        };
        await resultsWindow.ShowDialog(this);
    }

}