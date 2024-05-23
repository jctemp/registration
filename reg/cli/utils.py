import tomlkit


def serialize_data(config, name="config.toml"):
    # Serialize to TOML
    with open(f"{name}", "w") as toml_file:
        tomlkit.dump(config, toml_file)


def deserialize_toml(name="config.toml"):
    # Deserialize from TOML
    with open(f"{name}", "r") as toml_file:
        config_toml = tomlkit.load(toml_file)
    return config_toml
