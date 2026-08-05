"""Per-exercise fixtures and checks.

One module per exercise, named after it: `fast.labs.abliteration`, `fast.labs.distillation`.
Each module holds both the shared resources a lab needs (data, fixtures, plotting) and the
checks that grade it, so a notebook needs a single import:

    from fast.labs import abliteration as lab

    harmful, harmless = lab.activation_pairs()
    lab.check_difference_in_means(difference_in_means)

Keeping fixtures beside checks is deliberate — they usually need the same data, and a fixture
that drifts from what the check expects is a lab that fails for the wrong reason.
"""
