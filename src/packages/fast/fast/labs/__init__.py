"""Per-lab fixtures and checks, nested to mirror the lab directories under `src/`.

One module per lab, named and placed like the lab it serves (with hyphens turned into
underscores, since module names can't contain hyphens):

    src/day1-models/output-distributions/  ->  fast.labs.day1_models.output_distributions
    src/examples/template/                 ->  fast.labs.examples.template

Each module holds both the shared resources a lab needs (data, fixtures, plotting) and the
checks that grade it, so a notebook needs a single import:

    from fast.labs.day1_models import output_distributions as lab

    logits = lab.logits_for(model, tokenizer, "The capital of France is")
    lab.check_next_token_probs(next_token_probs)

Keeping fixtures beside checks is deliberate: they usually need the same data, and a fixture
that drifts from what the check expects is a lab that fails for the wrong reason.
"""
