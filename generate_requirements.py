import os


reqs = ["dill", "nltk"]

optional_reqs = {
    "nn": ["tensorflow"],
    "pandas": ["pandas", "pyarrow", "numpy", "scikit-learn"],
    "dev": ["black", "rich", "matplotlib"],
}

full = reqs + [item for sublist in optional_reqs.values() for item in sublist]


def generate_requirements():
    with open("requirements.txt", "w") as f:
        # f.write("\n".join(full))
        f.write(".[full]\n")

    if not os.path.exists("requirements"):
        os.makedirs("requirements")

    # minimal
    with open("requirements/minimal.txt", "w") as f:
        f.write("\n".join(reqs))

    for key, value in optional_reqs.items():
        with open(f"requirements/{key}.txt", "w") as f:
            f.write("\n".join(value))

    # full
    with open("requirements/full.txt", "w") as f:
        f.write("\n".join(full))


if __name__ == "__main__":
    generate_requirements()
