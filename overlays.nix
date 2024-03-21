final: prev: {
  pythonPackagesExtensions =
    prev.pythonPackagesExtensions
    ++ [
      (python-final: python-prev: {
        torch = python-prev.torch-bin;

        torchvision = python-prev.torchvision-bin;

        torchaudio = python-prev.torchaudio-bin;

        tensorflow = python-prev.tensorflow-bin;

        torchmetrics = python-prev.torchmetrics.override {
          torch = python-final.torch;
        };

        tensorboardx = python-prev.tensorboardx.override {
          torch = python-final.torch;
        };

        pytorch-lightning = python-prev.pytorch-lightning.override {
          torch = python-final.torch;
        };

        torchio = python-prev.pytorch-lightning.override {
          torch = python-final.torch;
        };

        monai = python-prev.pytorch-lightning.override {
          torch = python-final.torch;
        };
      })
    ];
}
