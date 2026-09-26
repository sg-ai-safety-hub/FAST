# %% [markdown]
# # Day 0: Smoke test
#
# Run this before Day 1. It confirms four things: you have a GPU runtime, the shared
# `fast` package installs, a model downloads, and generation works. It takes about
# two minutes and downloads roughly 300 MB.
#
# If any cell fails, bring the error to setup: an environment problem found today is
# a non-event, the same problem found on Wednesday morning costs you a lab.
#
# **Before you start:** Runtime > Change runtime type > T4 GPU.

# %%
# Installs the lab package on Colab; skipped when it's already importable (e.g. a local editable install).
try:
    import fast  # noqa: F401
except ImportError:
    # %pip install -q git+https://github.com/sg-ai-safety-hub/FAST.git@main#subdirectory=src/packages/fast
    pass

# %%
from fast.colab import setup

setup()

# %% [markdown]
# ## Load a model
#
# `smol-135m` is the smallest model we use: it exists so this test is quick. The labs
# themselves run 1-3B models.

# %%
from fast.models import chat, load_model

model, tokenizer = load_model("smol-135m")
print(f"loaded {model.config._name_or_path} · {model.num_parameters() / 1e6:.0f}M parameters")

# %%
print(chat(model, tokenizer, "In one sentence: what is a system card?"))

# %% [markdown]
# ## Checkpointing
#
# Colab runtimes disconnect. Labs that train anything write checkpoints to your Drive
# so a dropped runtime costs you minutes rather than the session. This mounts it.

# %%
from fast.colab import mount_drive

workdir = mount_drive()
print(f"working directory: {workdir}")

# %% [markdown]
# If you got here without an error, you're set for Day 1.
