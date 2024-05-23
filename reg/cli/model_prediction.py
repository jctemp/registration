config = {}

print(f"{'=' * 5} Configuration summary {'=' * 92}")
print(f"")
for key, value in config.items():
    print(f"{key:<25} = {value}")
print(f"")
print("=" * 120)

raise RuntimeError("Prediction not implemented.")

# TODO: accept a globbing string to match multiple mat files

# TODO: in the mat file path directory create subdirectory for output

# TODO: load model from ckpt

# TODO: apply data preprocessing

# TODO: make prediction

# TODO: run STN with flow and input mat

# TODO: save warped series to output path
